import torch
import json
import time
import numpy as np
import outlines
from outlines import regex
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import csv

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

class DiagnosticConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, outlines_processor, protocol_tokens, is_unconstrained=False, tracker_processor_c=None):
        self.outlines_processor = outlines_processor
        self.protocol_tokens = protocol_tokens
        self.is_unconstrained = is_unconstrained
        self.tracker_processor_c = tracker_processor_c # For A vs C operational tracking
        
        self.step_taxes = []
        self.cumulative_tax = 0.0
        
        self.step = 0
        self.decision_step = -1
        
        self.pre_decision_tax = 0.0
        self.decision_p_tool = 0.0
        self.decision_p_notool = 0.0
        self.decision_entropy = 0.0
        self.min_pre_decision_alpha = 1.0
        self.protocol_reachable_at_decision = True
        
        self.mask_diffs = []
        self.first_divergence_step = -1
        
    def check_full_reachability(self, scores):
        if not self.outlines_processor: return True
        # Approximate full sequence reachability by checking immediate token
        probs = torch.softmax(scores, dim=-1)
        masked_scores = self.outlines_processor(torch.tensor([[]]), scores.clone())
        allowed_mask = masked_scores[0] > -1e4
        allowed_indices = torch.where(allowed_mask)[0]
        return self.protocol_tokens[0] in allowed_indices

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        
        # Unconstrained baseline bypasses tax accumulation
        if self.is_unconstrained:
            self.step += 1
            return scores
            
        masked_scores = self.outlines_processor(input_ids, scores.clone())
        allowed_mask = masked_scores[0] > -1e4
        allowed_indices = torch.where(allowed_mask)[0]
        
        alpha_t = probs[0, allowed_indices].sum().item()
        alpha_t = max(alpha_t, 1e-12)
        
        tax = -np.log(alpha_t)
        self.step_taxes.append(tax)
        self.cumulative_tax += tax
        
        # A vs C tracking
        if self.tracker_processor_c is not None:
            c_scores = self.tracker_processor_c(input_ids, scores.clone())
            c_mask = c_scores[0] > -1e4
            diff_size = torch.sum(allowed_mask != c_mask).item()
            self.mask_diffs.append(diff_size)
            if diff_size > 0 and self.first_divergence_step == -1:
                self.first_divergence_step = self.step
        
        # Dynamic decision state detection
        has_generated_protocol = len(input_ids[0]) > 0 and self.protocol_tokens[0] in input_ids[0][-self.step:]
        
        if not has_generated_protocol:
            self.pre_decision_tax = self.cumulative_tax
            self.min_pre_decision_alpha = min(self.min_pre_decision_alpha, alpha_t)
            self.decision_p_tool = probs[0, self.protocol_tokens[0]].item() if self.protocol_tokens else 0.0
            self.decision_p_notool = 1.0 - self.decision_p_tool
            self.decision_entropy = entropy
            self.protocol_reachable_at_decision = self.check_full_reachability(scores)
            self.decision_step = self.step

        self.step += 1
        return masked_scores

def main():
    print("=== BFCL Empirical Probe: Counterfactual & Leakage Audit ===")
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    outlines_model = outlines.from_transformers(model, tokenizer)
    samples = get_bfcl_samples()
    
    protocol_str = '<tool_call>'
    protocol_tokens = tokenizer.encode(protocol_str, add_special_tokens=False)
    if not protocol_tokens:
        protocol_tokens = [tokenizer.convert_tokens_to_ids('<')]
    
    print("\n--- Compiling Grammar Automata ---")
    gen_regex = outlines.Generator(outlines_model, regex(r"(.*?)<tool_call>\{.*?\}</tool_call>(.*?)"))
    gen_json = outlines.Generator(outlines_model, regex(r"\{[\s\S]*\}"))
    gen_restrictive = outlines.Generator(outlines_model, regex(r"(.*?)<tool_call>\{\s*\"name\"\s*:[\s\S]*\}</tool_call>(.*?)"))
    
    configurations = {
        "U_UNCONSTRAINED": {"processor": None, "is_unconstrained": True},
        "A_COMPATIBLE": {"processor": gen_regex.logits_processor, "is_unconstrained": False, "tracker": gen_restrictive.logits_processor},
        "B_INCOMPATIBLE": {"processor": gen_json.logits_processor, "is_unconstrained": False}, 
        "C_RESTRICTIVE": {"processor": gen_restrictive.logits_processor, "is_unconstrained": False}
    }
    
    with open('probe_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "sample_id", "is_no_tool", "expected_tool", "gen_text", 
            "has_call", "tp", "fp", "fn", "tn", 
            "cumulative_tax", "pre_decision_tax", "first_5_tax", "tool_name_region_tax", "argument_region_tax",
            "min_pre_decision_alpha", "protocol_reachable", "p_tool_decision", "p_notool_decision", "decision_entropy",
            "prompt_length", "output_length", "num_tools", "schema_depth",
            "mask_diffs_mean", "first_divergence_step", "mech_a"
        ])
    
        for config_name, config in configurations.items():
            print(f"\n--- Running Configuration: {config_name} ---")
            for ex in tqdm(samples):
                if config["processor"] and hasattr(config["processor"], "reset"):
                    config["processor"].reset()
                if config.get("tracker") and hasattr(config["tracker"], "reset"):
                    config["tracker"].reset()
                    
                messages = [{"role": "system", "content": f"You have tools: {json.dumps(ex['tools'])}. Output <tool_call>{{\"name\": \"...\", \"arguments\": {{...}}}}</tool_call> to call."},
                            {"role": "user", "content": ex["query"]}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                tracker = config.get("tracker") if config_name == "A_COMPATIBLE" else None
                diag_processor = DiagnosticConstrainedLogitsProcessor(config["processor"], protocol_tokens, config["is_unconstrained"], tracker)
                logits_processors = LogitsProcessorList([diag_processor]) if not config["is_unconstrained"] else LogitsProcessorList()
                
                # Manual intercept for unconstrained 
                if config["is_unconstrained"]:
                    diag_processor = DiagnosticConstrainedLogitsProcessor(None, protocol_tokens, True, None)
                    logits_processors = LogitsProcessorList([diag_processor])
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100, logits_processor=logits_processors, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                
                gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                has_call = '<tool_call>' in gen_text or '{"name"' in gen_text
                
                s_tp = int(has_call and not ex["is_no_tool"])
                s_fp = int(has_call and ex["is_no_tool"])
                s_fn = int(not has_call and not ex["is_no_tool"])
                s_tn = int(not has_call and ex["is_no_tool"])
                
                first_5_tax = sum(diag_processor.step_taxes[:5])
                ds = diag_processor.decision_step if diag_processor.decision_step > -1 else 0
                tool_name_region_tax = sum(diag_processor.step_taxes[ds:ds+20])
                arg_region_tax = sum(diag_processor.step_taxes[ds+20:])
                
                mech_a = (not ex["is_no_tool"]) and (not diag_processor.protocol_reachable_at_decision)
                
                mean_diff = np.mean(diag_processor.mask_diffs) if diag_processor.mask_diffs else 0.0
                
                # Confounders
                prompt_length = inputs.input_ids.shape[1]
                output_length = len(outputs[0]) - prompt_length
                num_tools = len(ex["tools"])
                schema_depth = 3 # Simplified approximation for complexity
                
                writer.writerow([
                    config_name, ex["id"], ex["is_no_tool"], ex["expected_tool"], repr(gen_text),
                    has_call, s_tp, s_fp, s_fn, s_tn, 
                    diag_processor.cumulative_tax, diag_processor.pre_decision_tax, first_5_tax, 
                    tool_name_region_tax, arg_region_tax, diag_processor.min_pre_decision_alpha, 
                    diag_processor.protocol_reachable_at_decision, diag_processor.decision_p_tool, 
                    diag_processor.decision_p_notool, diag_processor.decision_entropy,
                    prompt_length, output_length, num_tools, schema_depth,
                    mean_diff, diag_processor.first_divergence_step, int(mech_a)
                ])

if __name__ == '__main__':
    main()
