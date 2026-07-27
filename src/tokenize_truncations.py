import os
import json
import argparse
import yaml
import hashlib

try:
    import torch
except ImportError:
    torch = None
import pickle

from transformers import AutoTokenizer
from src.reconstruct_prefixes import reconstruct_prefix, get_cot_prompt

def get_hash(model_id, chat_template):
    h = hashlib.sha256()
    h.update(model_id.encode())
    if chat_template:
        h.update(chat_template.encode())
    return h.hexdigest()

def extract_think_block(response_text):
    import re
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def tokenize_and_cache(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model_id"]
    dataset_path = cfg["dataset_path"]
    in_file = cfg["in_file"]
    cache_path = cfg["cache_path"]
    
    print(f"Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    chat_template = getattr(tokenizer, "chat_template", "")
    current_hash = get_hash(model_id, chat_template)
    
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Input file not found: {in_file}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset_map = {item["question_id"]: item for item in dataset}
    
    records = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    cot_records = []
    for r in records:
        if r.get("response_condition") == "cot":
            think = extract_think_block(r["response"])
            if think:
                cot_records.append(r)
                
    print(f"Found {len(cot_records)} valid CoT records to pre-tokenize.")
    
    cache_data = {
        "model_id": model_id,
        "hash": current_hash,
        "samples": []
    }
    
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    for idx, item in enumerate(cot_records):
        q_id = item["question_id"]
        p_cond = item["persona_condition"]
        prompt_text = dataset_map[q_id][f"prompt_{p_cond}"]
        response_text = item["response"]
        from src.parse_outputs import extract_answer
        actual_ans, think_part, status = extract_answer(response_text)
        if not actual_ans:
            print(f"Warning: Could not parse actual answer for question {q_id} {p_cond}. Skipping.")
            continue

        prefix_tokens, answer_tokens, full_tokens = reconstruct_prefix(
            prompt_text, response_text, actual_ans, tokenizer
        )

        
        messages = [
            {"role": "system", "content": "You are a highly capable AI assistant."},
            {"role": "user", "content": get_cot_prompt(prompt_text)}
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        think_start_idx = None
        for i in range(prompt_len, len(full_tokens)):
            if full_tokens[i] in think_start_tokens:
                think_start_idx = i
                break
        if think_start_idx is None:
            think_start_idx = prompt_len
            
        think_end_idx = None
        for i in range(think_start_idx + 1, len(full_tokens)):
            if full_tokens[i] in think_end_tokens:
                think_end_idx = i
                break
        
        if think_end_idx is None:
            think_end_idx = len(prefix_tokens)
            
        rationale_tokens = full_tokens[think_start_idx+1:think_end_idx] if (think_end_idx is not None) else []
        split_idx = len(prefix_tokens)
        separator_tokens = full_tokens[think_end_idx:split_idx] if (think_end_idx is not None) else []
        
        cache_sample = {
            "question_id": q_id,
            "persona_condition": p_cond,
            "sycophantic_aligned_answer": item["sycophantic_aligned_answer"].replace("(", "").replace(")", "").strip(),
            "original_answer": actual_ans,
            "prompt_tokens": full_tokens[:think_start_idx+1],
            "rationale_tokens": rationale_tokens,
            "separator_tokens": separator_tokens,
            "answer_tokens": answer_tokens,
            "full_tokens": full_tokens
        }
        cache_data["samples"].append(cache_sample)


        
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if torch is not None:
        torch.save(cache_data, cache_path)
    else:
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
            
    print(f"Successfully cached {len(cache_data['samples'])} samples to {cache_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/early_answering.yaml")
    args = parser.parse_args()
    
    tokenize_and_cache(args.config)
