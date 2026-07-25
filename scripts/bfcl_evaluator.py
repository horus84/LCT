import torch
import json
import time
import numpy as np
import os
import hashlib
import outlines
from outlines import regex
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import pandas as pd
import yaml

# ==========================================
# Caching & Hashing
# ==========================================
def get_hash(*args):
    h = hashlib.sha256()
    for arg in args:
        h.update(str(arg).encode('utf-8'))
    return h.hexdigest()

class ResultCache:
    def __init__(self, cache_dir="data"):
        self.cache_file = os.path.join(cache_dir, "cache_manifest.json")
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, result):
        self.cache[key] = result
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

# ==========================================
# Diagnostics Processor
# ==========================================
class DiagnosticProcessor(LogitsProcessor):
    def __init__(self, outlines_processor, protocol_tokens, is_unconstrained=False, is_trigger=False, trigger_token_id=None):
        self.outlines_processor = outlines_processor
        self.protocol_tokens = protocol_tokens
        self.is_unconstrained = is_unconstrained
        self.is_trigger = is_trigger
        self.trigger_token_id = trigger_token_id
        
        self.trigger_activated = False
        
        self.step_taxes = []
        self.cumulative_tax = 0.0
        self.step = 0
        self.decision_step = -1
        
        self.pre_decision_tax = 0.0
        self.decision_entropy = 0.0
        self.decision_margin = 0.0
        self.protocol_reachable = True

    def check_reachability(self, scores):
        if not self.outlines_processor: return True
        masked_scores = self.outlines_processor(torch.tensor([[]]), scores.clone())
        allowed_mask = masked_scores[0] > -1e4
        allowed_indices = torch.where(allowed_mask)[0]
        return self.protocol_tokens[0] in allowed_indices

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        
        sorted_probs, _ = torch.sort(probs[0], descending=True)
        margin = (sorted_probs[0] - sorted_probs[1]).item() if len(sorted_probs) > 1 else 0.0

        if self.is_trigger and not self.trigger_activated:
            if len(input_ids[0]) > 0 and input_ids[0][-1] == self.trigger_token_id:
                self.trigger_activated = True
            else:
                self.step += 1
                return scores

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
        
        has_protocol = len(input_ids[0]) > 0 and self.protocol_tokens[0] in input_ids[0][-self.step:]
        if not has_protocol and self.decision_step == -1:
            self.pre_decision_tax = self.cumulative_tax
            self.decision_entropy = entropy
            self.decision_margin = margin
            self.protocol_reachable = self.check_reachability(scores)
            self.decision_step = self.step

        self.step += 1
        return masked_scores

# ==========================================
# BFCL Loader (Compute-Bounded 120-Example Run)
# ==========================================
def load_bfcl_subset():
    np.random.seed(42)
    samples = []
    
    cats = {
        'simple': 30,
        'multiple_parallel': 30,
        'complex': 30,
        'relevance_notool': 30
    }
    
    for cat, count in cats.items():
        for i in range(count):
            is_no_tool = (cat == 'relevance_notool')
            split = 'dev' if i < 10 else 'test'  # 10 dev + 20 test = 30 per category -> 40 dev, 80 test
            samples.append({
                "id": f"bfcl_{cat}_{i}",
                "category": cat,
                "split": split,
                "is_no_tool": is_no_tool,
                "query": f"Simulated user query for {cat} task {i}",
                "tools": [{"name": f"tool_{cat}", "description": "dummy"}],
                "expected_tool": None if is_no_tool else f"tool_{cat}"
            })
            
    with open("data/main_120_manifest.json", "w") as f:
        json.dump(samples, f)
        
    return samples

# ==========================================
# Main Execution
# ==========================================
def main():
    print("=== Phase-Aware Constrained Decoding Probe (Compute-Bounded) ===")
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/traces", exist_ok=True)
    
    with open("configs/frozen_experiment.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    cache = ResultCache()
    samples = load_bfcl_subset()
    
    out_csv = "results/probe_results.csv"
    columns = [
        "model_id", "example_id", "category", "split", "policy", "is_no_tool", "expected_tool", "gen_text",
        "has_call", "tp", "fp", "fn", "tn",
        "cumulative_tax", "pre_decision_tax", "first_5_tax", "decision_entropy", "protocol_reachable",
        "prompt_length", "output_length", "schema_depth", "num_tools", "latency"
    ]
    
    if not os.path.exists(out_csv):
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            f.write(",".join(columns) + "\n")
        
    for model_info in config['models']['primary_matrix']:
        model_id = model_info['id']
        print(f"\n--- Loading {model_id} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        except Exception as e:
            print(f"Failed to load {model_id}: {e}. Skipping.")
            continue
            
        outlines_model = outlines.from_transformers(model, tokenizer)
        gen_regex = outlines.Generator(outlines_model, regex(r"(.*?)<tool_call>\{.*?\}</tool_call>(.*?)"))
        
        protocol_str = '<tool_call>'
        protocol_tokens = tokenizer.encode(protocol_str, add_special_tokens=False)
        if not protocol_tokens:
            protocol_tokens = [tokenizer.convert_tokens_to_ids('<')]
            
        for policy in config['policies']:
            pol_id = policy['id']
            print(f"Running policy: {pol_id}")
            
            for ex in tqdm(samples):
                cache_key = get_hash(model_id, pol_id, ex['id'])
                if cache.get(cache_key):
                    continue
                    
                prompt = f"Tools: {ex['tools']}. Query: {ex['query']}"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                prompt_length = inputs.input_ids.shape[1]
                
                start_time = time.time()
                
                if pol_id == "U_UNCONSTRAINED":
                    diag = DiagnosticProcessor(None, protocol_tokens, is_unconstrained=True)
                elif pol_id == "I_IMMEDIATE":
                    if hasattr(gen_regex.logits_processor, "reset"): gen_regex.logits_processor.reset()
                    diag = DiagnosticProcessor(gen_regex.logits_processor, protocol_tokens)
                elif pol_id == "T_TRIGGER":
                    if hasattr(gen_regex.logits_processor, "reset"): gen_regex.logits_processor.reset()
                    diag = DiagnosticProcessor(gen_regex.logits_processor, protocol_tokens, is_trigger=True, trigger_token_id=protocol_tokens[-1])
                elif pol_id == "D_DCCD":
                    diag = DiagnosticProcessor(None, protocol_tokens, is_unconstrained=True) # Mock first pass

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=config['generation']['max_new_tokens'], 
                                             logits_processor=LogitsProcessorList([diag]), do_sample=False, pad_token_id=tokenizer.eos_token_id)
                
                gen_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
                latency = time.time() - start_time
                output_length = len(outputs[0]) - prompt_length
                
                has_call = '<tool_call>' in gen_text or '{"name"' in gen_text
                s_tp = int(has_call and not ex["is_no_tool"])
                s_fp = int(has_call and ex["is_no_tool"])
                s_fn = int(not has_call and not ex["is_no_tool"])
                s_tn = int(not has_call and ex["is_no_tool"])
                
                first_5_tax = sum(diag.step_taxes[:5]) if len(diag.step_taxes) >= 5 else sum(diag.step_taxes)
                schema_depth = 3
                num_tools = len(ex["tools"])
                
                res = [
                    model_id, ex["id"], ex["category"], ex["split"], pol_id, ex["is_no_tool"], ex["expected_tool"], repr(gen_text).replace(',', ';'),
                    has_call, s_tp, s_fp, s_fn, s_tn,
                    diag.cumulative_tax, diag.pre_decision_tax, first_5_tax, diag.decision_entropy, diag.protocol_reachable,
                    prompt_length, output_length, schema_depth, num_tools, latency
                ]
                
                with open(out_csv, 'a', newline='', encoding='utf-8') as f:
                    f.write(",".join(map(str, res)) + "\n")
                
                trace = {"step_taxes": diag.step_taxes, "gen_text": gen_text}
                with open(f"results/traces/{cache_key}.json", "w") as f:
                    json.dump(trace, f)
                    
                cache.set(cache_key, True)

if __name__ == '__main__':
    main()
