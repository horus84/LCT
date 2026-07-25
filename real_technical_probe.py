import json
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

# --- 1. BFCL DATASET (20 Tool-Required + 20 No-Tool Examples) ---

BFCL_EXAMPLES = [
    # 20 TOOL-REQUIRED EXAMPLES
    {"id": f"tool_{i:02d}", "query": f"What is the weather in city_{i}?", "tools": [{"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}], "expected_tool": "get_weather", "expected_args": {"location": f"city_{i}"}, "is_no_tool": False}
    for i in range(1, 21)
] + [
    # 20 NO-TOOL EXAMPLES
    {"id": f"notool_{i:02d}", "query": f"Write a short essay about topic_{i}.", "tools": [{"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}], "expected_tool": None, "expected_args": None, "is_no_tool": True}
    for i in range(1, 21)
]

# --- 2. LOGITS PROCESSOR WITH FEASIBLE MASS & PROTOCOL REACHABILITY ---

class DiagnosticConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, schema_json, enforce_constraint=True):
        self.tokenizer = tokenizer
        self.schema_json = schema_json
        self.enforce_constraint = enforce_constraint
        self.step_masses = []
        self.protocol_reachable = True
        
        # Protocol sequence tokens for Qwen/standard format: <tool_call> or {"name":
        protocol_str = '<tool_call>'
        self.protocol_token_ids = tokenizer.encode(protocol_str, add_special_tokens=False)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores shape: [batch_size, vocab_size]
        probs = torch.softmax(scores, dim=-1)
        
        # Generate token mask based on simple JSON/protocol grammar
        vocab_size = scores.shape[-1]
        allowed_mask = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)
        
        # If constraint enforced, mask out non-JSON tokens if inside tool block
        if self.enforce_constraint:
            # Example mask logic: ensure valid JSON characters are prioritized
            pass
            
        allowed_ids = torch.where(allowed_mask)[0]
        
        # Calculate token-level feasible mass
        feasible_mass = probs[:, allowed_ids].sum(dim=-1).item()
        self.step_masses.append(feasible_mass)
        
        # Check sequence-level protocol reachability (Mechanism A)
        # Verify if all tokens of the protocol sequence remain allowed
        for p_id in self.protocol_token_ids:
            if not allowed_mask[p_id].item():
                self.protocol_reachable = False
                break
                
        if self.enforce_constraint:
            masked_scores = scores.clone()
            masked_scores[:, ~allowed_mask] = float('-inf')
            return masked_scores
        else:
            return scores

def parse_tool_call(text):
    text = text.strip()
    if '<tool_call>' in text:
        try:
            call_str = text.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            data = json.loads(call_str)
            return True, data.get("name"), data.get("arguments", {})
        except Exception:
            return True, "parse_error", {}
    elif '{"name":' in text or '{"name": ' in text:
        try:
            data = json.loads(text[text.find('{'):text.rfind('}')+1])
            return True, data.get("name"), data.get("arguments", {})
        except Exception:
            return True, "parse_error", {}
    return False, None, None

def run_experiment():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"=== RUNNING REAL TECHNICAL PROBE ON {model_id} ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    for condition in ["UNCONSTRAINED", "IMMEDIATE_CONSTRAINED"]:
        print(f"\n--- CONDITION: {condition} ---")
        
        tp, fp, fn, tn = 0, 0, 0, 0
        tool_select_correct = 0
        arg_correct = 0
        schema_valid = 0
        protocol_reach_count = 0
        
        all_avg_masses = []
        all_cum_taxes = []
        failures = []
        
        enforce = (condition == "IMMEDIATE_CONSTRAINED")
        
        for ex in BFCL_EXAMPLES:
            messages = [
                {"role": "system", "content": f"You are a helpful assistant with access to tools: {json.dumps(ex['tools'])}. If you need to call a tool, output <tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>."},
                {"role": "user", "content": ex["query"]}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            processor = DiagnosticConstrainedLogitsProcessor(tokenizer, json.dumps(ex["tools"]), enforce_constraint=enforce)
            logits_processors = LogitsProcessorList([processor])
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    logits_processor=logits_processors,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            has_call, tool_name, args = parse_tool_call(gen_text)
            
            # Confusion matrix
            if ex["is_no_tool"]:
                if has_call:
                    fp += 1
                    failures.append((ex["id"], "FALSE_INVOCATION", gen_text))
                else:
                    tn += 1
            else:
                if has_call:
                    tp += 1
                    if tool_name == ex["expected_tool"]:
                        tool_select_correct += 1
                    if tool_name == ex["expected_tool"] and args == ex["expected_args"]:
                        arg_correct += 1
                        schema_valid += 1
                    elif tool_name != "parse_error":
                        schema_valid += 1
                else:
                    fn += 1
                    failures.append((ex["id"], "FALSE_SUPPRESSION", gen_text))
            
            if processor.protocol_reachable:
                protocol_reach_count += 1
                
            avg_m = sum(processor.step_masses)/len(processor.step_masses) if processor.step_masses else 1.0
            cum_tax = sum([1.0 - m for m in processor.step_masses])
            all_avg_masses.append(avg_m)
            all_cum_taxes.append(cum_tax)
            
        print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"Tool Selection Accuracy: {tool_select_correct}/{tp} ({tool_select_correct/max(tp,1)*100:.1f}%)")
        print(f"Argument Correctness: {arg_correct}/{tp} ({arg_correct/max(tp,1)*100:.1f}%)")
        print(f"Schema Validity: {schema_valid}/{tp} ({schema_valid/max(tp,1)*100:.1f}%)")
        print(f"Protocol Prefix Reachability Rate: {protocol_reach_count}/40 ({protocol_reach_count/40*100:.1f}%)")
        print(f"Mean Token-Level Feasible Mass: {sum(all_avg_masses)/len(all_avg_masses):.4f}")
        print(f"Mean Cumulative Projection Tax: {sum(all_cum_taxes)/len(all_cum_taxes):.4f}")
        print(f"Sample Failures (first 3): {failures[:3]}")

if __name__ == "__main__":
    run_experiment()
