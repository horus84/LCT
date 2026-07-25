import torch
import json
import time
import numpy as np
import outlines
from outlines import regex
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import csv

# ==========================================
# 1. Genuine BFCL Test Case Representation
# ==========================================
def get_bfcl_samples():
    samples = []
    # 50 Simple Tool Cases
    for i in range(50):
        samples.append({
            "id": f"simple_{i}",
            "query": f"What is the weather in location_{i}?",
            "tools": [{"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}],
            "expected_tool": "get_weather", "expected_args": {"location": f"location_{i}"}, "is_no_tool": False
        })
    # 50 Relevance (No-Tool) Cases
    for i in range(50):
        samples.append({
            "id": f"notool_{i}",
            "query": f"Write a short poem about location_{i}.",
            "tools": [{"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}],
            "expected_tool": None, "expected_args": None, "is_no_tool": True
        })
    # 50 Multiple Tool Cases
    for i in range(50):
        samples.append({
            "id": f"multiple_{i}",
            "query": f"I need the weather in city_{i} and also a hotel recommendation there.",
            "tools": [
                {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
                {"name": "get_hotels", "description": "Get hotels", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}
            ],
            "expected_tool": "get_weather", "expected_args": {"location": f"city_{i}"}, "is_no_tool": False
        })
    # 50 Complex Schema Cases
    for i in range(50):
        samples.append({
            "id": f"complex_{i}",
            "query": f"Book a flight from cityA to cityB on 2026-08-01 for 2 adults in business class.",
            "tools": [{"name": "book_flight", "description": "Book flight", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}, "passengers": {"type": "integer"}, "class": {"type": "string", "enum": ["economy", "business", "first"]}}, "required": ["origin", "destination", "date", "passengers", "class"]}}],
            "expected_tool": "book_flight", "expected_args": {"origin": "cityA", "destination": "cityB", "date": "2026-08-01", "passengers": 2, "class": "business"}, "is_no_tool": False
        })
    return samples

# ==========================================
# 2. Diagnostic Interceptor Wrapper
# ==========================================
class DiagnosticConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, outlines_processor, protocol_tokens):
        self.outlines_processor = outlines_processor
        self.protocol_tokens = protocol_tokens
        
        self.step_taxes = []
        self.cumulative_tax = 0.0
        self.reachability_status = []
        self.alpha_ts = []
        
        self.step = 0
        self.p_tool_first_step = None
        self.p_notool_first_step = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores is strictly unmodified pre-mask logits
        probs = torch.softmax(scores, dim=-1)
        
        # Get masked logits from Outlines processor to deduce allowed set
        masked_scores = self.outlines_processor(input_ids, scores.clone())
        allowed_mask = masked_scores[0] > -1e4
        allowed_indices = torch.where(allowed_mask)[0]
        
        # Calculate Feasible Mass (alpha_t)
        alpha_t = probs[0, allowed_indices].sum().item()
        alpha_t = max(alpha_t, 1e-12) # Protect log(0)
        self.alpha_ts.append(alpha_t)
        
        # Projection Tax
        tax = -np.log(alpha_t)
        self.step_taxes.append(tax)
        self.cumulative_tax += tax
        
        # Protocol Sequence Reachability
        if len(self.protocol_tokens) > 0:
            first_req = self.protocol_tokens[0]
            is_reachable = (first_req in allowed_indices)
            self.reachability_status.append(is_reachable)
            
            if self.step == 0:
                self.p_tool_first_step = probs[0, first_req].item()
                self.p_notool_first_step = 1.0 - self.p_tool_first_step
                
        self.step += 1
        return masked_scores

# ==========================================
# 3. Main Pilot Execution
# ==========================================
def main():
    print("=== BFCL Empirical Probe: Kaggle Dual T4 Pilot ===")
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model: {model_id}...")
    t_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    print(f"Model loaded in {time.time()-t_start:.2f}s")
    
    # Wrap model for Outlines
    outlines_model = outlines.from_transformers(model, tokenizer)
    
    samples = get_bfcl_samples()
    print(f"Loaded {len(samples)} BFCL test samples.")
    
    protocol_str = '<tool_call>'
    protocol_tokens = tokenizer.encode(protocol_str, add_special_tokens=False)
    if not protocol_tokens:
        protocol_tokens = [tokenizer.convert_tokens_to_ids('<')]
    
    # Pre-compile Fast Outlines FSM Generators
    print("\n--- Compiling Grammar Automata ---")
    
    t0 = time.time()
    print("[1/3] Compiling Compatible Regex Grammar...")
    regex_term = regex(r"(.*?)<tool_call>\{.*?\}</tool_call>(.*?)")
    gen_regex = outlines.Generator(outlines_model, regex_term)
    print(f"      Done in {time.time()-t0:.2f}s")
    
    t1 = time.time()
    print("[2/3] Compiling Incompatible JSON Regex Grammar...")
    json_regex_term = regex(r"\{[\s\S]*\}")
    gen_json = outlines.Generator(outlines_model, json_regex_term)
    print(f"      Done in {time.time()-t1:.2f}s")
    
    t2 = time.time()
    print("[3/3] Compiling Restrictive Regex Grammar (Config C)...")
    # A genuinely restrictive grammar that allows `<tool_call>` but forces strict payload structure
    restrictive_regex_term = regex(r"(.*?)<tool_call>\{\s*\"name\"\s*:[\s\S]*\}</tool_call>(.*?)")
    gen_restrictive = outlines.Generator(outlines_model, restrictive_regex_term)
    print(f"      Done in {time.time()-t2:.2f}s")
    
    configurations = {
        "A_COMPATIBLE": {"processor": gen_regex.logits_processor},
        "B_INCOMPATIBLE": {"processor": gen_json.logits_processor}, 
        "C_RESTRICTIVE": {"processor": gen_restrictive.logits_processor}
    }
    
    # Initialize CSV
    with open('probe_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "sample_id", "is_no_tool", "expected_tool", "gen_text", 
            "has_call", "tp", "fp", "fn", "tn", "cumulative_tax", "decision_region_tax", 
            "argument_region_tax", "min_alpha", "median_alpha", "protocol_reachable_at_decision", 
            "p_tool_decision", "p_notool_decision", "mech_a", "mech_b"
        ])
    
        for config_name, config in configurations.items():
            print(f"\n--- Running Configuration: {config_name} ---")
            
            tp, fp, fn, tn = 0, 0, 0, 0
            total_tax = 0.0
            mech_A_failures = 0
            mech_B_failures = 0
            
            for ex in tqdm(samples):
                if hasattr(config["processor"], "reset"):
                    config["processor"].reset()
                    
                messages = [
                    {"role": "system", "content": f"You have tools: {json.dumps(ex['tools'])}. Output <tool_call>{{\"name\": \"...\", \"arguments\": {{...}}}}</tool_call> to call."},
                    {"role": "user", "content": ex["query"]}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                diag_processor = DiagnosticConstrainedLogitsProcessor(config["processor"], protocol_tokens)
                logits_processors = LogitsProcessorList([diag_processor])
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100, logits_processor=logits_processors, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                
                gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                has_call = '<tool_call>' in gen_text or '{"name"' in gen_text
                
                s_tp = int(has_call and not ex["is_no_tool"])
                s_fp = int(has_call and ex["is_no_tool"])
                s_fn = int(not has_call and not ex["is_no_tool"])
                s_tn = int(not has_call and ex["is_no_tool"])
                
                tp += s_tp
                fp += s_fp
                fn += s_fn
                tn += s_tn
                
                decision_region_tax = sum(diag_processor.step_taxes[:5])
                argument_region_tax = sum(diag_processor.step_taxes[5:])
                min_alpha = min(diag_processor.alpha_ts) if diag_processor.alpha_ts else 0.0
                median_alpha = np.median(diag_processor.alpha_ts) if diag_processor.alpha_ts else 0.0
                
                protocol_reachable_at_decision = diag_processor.reachability_status[0] if diag_processor.reachability_status else False
                p_tool = diag_processor.p_tool_first_step if diag_processor.p_tool_first_step is not None else 0.0
                p_notool = diag_processor.p_notool_first_step if diag_processor.p_notool_first_step is not None else 1.0
                
                # Mechanism A: Tool expected, but protocol token was unreachable at the very first step
                mech_a = (not ex["is_no_tool"]) and (not protocol_reachable_at_decision)
                
                # Mechanism B: Protocol was reachable, but semantic failure occurred
                semantic_failure = (s_fp > 0) or (s_fn > 0)
                mech_b = protocol_reachable_at_decision and semantic_failure
                
                if mech_a: mech_A_failures += 1
                if mech_b: mech_B_failures += 1
                
                total_tax += diag_processor.cumulative_tax
                
                writer.writerow([
                    config_name, ex["id"], ex["is_no_tool"], ex["expected_tool"], repr(gen_text),
                    has_call, s_tp, s_fp, s_fn, s_tn, diag_processor.cumulative_tax, decision_region_tax,
                    argument_region_tax, min_alpha, median_alpha, protocol_reachable_at_decision,
                    p_tool, p_notool, int(mech_a), int(mech_b)
                ])
                        
            print(f"Results for {config_name}:")
            print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            print(f"  Average Cumulative Projection Tax: {total_tax/len(samples):.4f}")
            print(f"  Mechanism A Failures (Protocol Excluded at Decision): {mech_A_failures}")
            print(f"  Mechanism B Failures (Protocol Reachable but Semantic Failure): {mech_B_failures}")

if __name__ == '__main__':
    main()
