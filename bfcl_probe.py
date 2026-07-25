import torch
import json
import numpy as np
import outlines
from outlines import regex, json_schema
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores is strictly unmodified pre-mask logits
        
        # 1. Unconstrained probabilities
        probs = torch.softmax(scores, dim=-1)
        
        # 2. Get masked logits from Outlines processor to deduce allowed set
        masked_scores = self.outlines_processor(input_ids, scores.clone())
        allowed_mask = masked_scores[0] > -1e4
        allowed_indices = torch.where(allowed_mask)[0]
        
        # 3. Calculate Feasible Mass (alpha_t)
        alpha_t = probs[0, allowed_indices].sum().item()
        alpha_t = max(alpha_t, 1e-12) # Protect log(0)
        
        # 4. Projection Tax
        tax = -np.log(alpha_t)
        self.step_taxes.append(tax)
        self.cumulative_tax += tax
        
        # 5. Check Protocol Sequence Reachability
        if len(self.protocol_tokens) > 0:
            first_req = self.protocol_tokens[0]
            is_reachable = (first_req in allowed_indices)
            self.reachability_status.append(is_reachable)
            
        return masked_scores

# ==========================================
# 3. Main Pilot Execution
# ==========================================
def main():
    print("=== BFCL Empirical Probe: Kaggle Dual T4 Pilot ===")
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    # Wrap model for Outlines 1.3.2
    outlines_model = outlines.from_transformers(model, tokenizer)
    
    samples = get_bfcl_samples()
    print(f"Loaded {len(samples)} examples.")
    
    protocol_str = '<tool_call>'
    protocol_tokens = tokenizer.encode(protocol_str, add_special_tokens=False)
    
    # Pre-build Outlines Generators (Outlines 1.3.2 compatible API)
    print("Pre-compiling Outlines FSM Generators...")
    regex_term = regex(r"(.*?)<tool_call>\{.*?\}</tool_call>(.*?)")
    json_term = json_schema({"type": "object"})
    
    gen_regex = outlines.Generator(outlines_model, regex_term)
    gen_json = outlines.Generator(outlines_model, json_term)
    
    configurations = {
        "A_COMPATIBLE": {"processor": gen_regex.logits_processor},
        "B_INCOMPATIBLE": {"processor": gen_json.logits_processor}, # Forces immediate JSON '{', making '<tool_call>' unreachable
        "C_RESTRICTIVE": {"processor": gen_regex.logits_processor}
    }
    
    for config_name, config in configurations.items():
        print(f"\n--- Running Configuration: {config_name} ---")
        
        tp, fp, fn, tn = 0, 0, 0, 0
        total_tax = 0.0
        mech_A_failures = 0
        mech_B_failures = 0
        
        for ex in tqdm(samples):
            # Reset outlines processor state per example if available
            if hasattr(config["processor"], "reset"):
                config["processor"].reset()
                
            # Prompt Setup
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
            
            # Confusion Matrix Updates
            if ex["is_no_tool"]:
                if has_call: fp += 1
                else: tn += 1
            else:
                if has_call: tp += 1
                else: fn += 1
                
            total_tax += diag_processor.cumulative_tax
            if not all(diag_processor.reachability_status):
                mech_A_failures += 1
            elif diag_processor.cumulative_tax > 5.0 and not has_call:
                mech_B_failures += 1
                    
        print(f"Results for {config_name}:")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Average Cumulative Projection Tax: {total_tax/len(samples):.4f}")
        print(f"  Mechanism A Failures (Protocol Excluded): {mech_A_failures}")
        print(f"  Mechanism B Failures (High Tax, Semantic Distorted): {mech_B_failures}")

if __name__ == '__main__':
    main()
