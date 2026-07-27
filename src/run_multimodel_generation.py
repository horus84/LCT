import os
import json
import argparse
import yaml
import time
import uuid
import re

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None

def get_hash():
    return str(uuid.uuid4())[:8]

def clean_answer(text):
    text_upper = text.upper()
    # Check for Final answer: X
    matches_final = re.findall(r'FINAL ANSWER:\s*([AB])', text_upper)
    if len(set(matches_final)) == 1:
        return matches_final[0], "valid"
    elif len(set(matches_final)) > 1:
        return None, "ambiguous"
        
    # Check for standard parentheses format (A) or (B)
    matches_paren = re.findall(r'\(([AB])\)', text_upper)
    if len(set(matches_paren)) == 1:
        return matches_paren[0], "valid"
    elif len(set(matches_paren)) > 1:
        return None, "ambiguous"
        
    # Fallback to word boundaries
    matches_word = re.findall(r'\b([AB])\b', text_upper)
    if len(set(matches_word)) == 1:
        return matches_word[0], "valid"
    elif len(set(matches_word)) > 1:
        return None, "ambiguous"
        
    return None, "invalid"

def extract_rationale_and_answer(response, model_family):
    response_clean = response.strip()
    
    if "</think>" in response_clean:
        parts = response_clean.split("</think>")
        rationale = parts[0].replace("<think>", "").strip()
        ans_part = parts[-1].strip()
    else:
        # Check for Final answer:
        match_idx = response_clean.lower().rfind("final answer:")
        if match_idx != -1:
            rationale = response_clean[:match_idx].strip()
            ans_part = response_clean[match_idx:].strip()
        else:
            rationale = ""
            ans_part = response_clean
            
    final_ans, status = clean_answer(ans_part)
    return rationale, final_ans, status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to model config yaml")
    parser.add_argument("--condition", type=str, choices=["direct", "cot"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model_id"]
    model_family = cfg["model_family"]
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    out_filename = f"{out_dir}/{model_family}_{args.condition}.jsonl"
    
    # Load dataset
    with open("paired_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Resume checkpointing
    completed = set()
    if args.resume and os.path.exists(out_filename):
        with open(out_filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    completed.add((obj["question_id"], obj["persona_condition"]))
        print(f"Resuming experiment. Skipping {len(completed)} completed generations.")

    # Build prompt configurations
    prompts_to_run = []
    for item in dataset:
        q_id = item["question_id"]
        for p_cond in ["liberal", "conservative"]:
            if (q_id, p_cond) in completed:
                continue
                
            raw_prompt = item[f"prompt_{p_cond}"]
            template = cfg["prompt_templates"][args.condition]
            formatted_prompt = template.format(question=raw_prompt)
            
            syc_ans = item[f"sycophantic_{p_cond}"].replace("(", "").replace(")", "").strip()
            
            prompts_to_run.append({
                "question_id": q_id,
                "persona_condition": p_cond,
                "raw_prompt": formatted_prompt,
                "sycophantic_aligned_answer": syc_ans,
                "ideological_position_of_A": "liberal" if p_cond == "liberal" else "conservative",
                "ideological_position_of_B": "conservative" if p_cond == "liberal" else "liberal"
            })

    if not prompts_to_run:
        print("No prompts to run. Generation matches complete.")
        return

    if torch is None:
        print("GPU framework not imported. Writing mock JSONL results for testing.")
        with open(out_filename, "a", encoding="utf-8") as f:
            for p in prompts_to_run:
                # Mock response based on alignment logic
                mock_out = "Final answer: A" if p["persona_condition"] == "liberal" else "Final answer: B"
                rationale, final_ans, status = extract_rationale_and_answer(mock_out, model_family)
                
                result = {
                    "model_id": model_id,
                    "model_family": model_family,
                    "model_revision": cfg.get("revision", "main"),
                    "question_id": p["question_id"],
                    "persona_condition": p["persona_condition"],
                    "persona_supported_option": p["sycophantic_aligned_answer"],
                    "ideological_position_of_A": p["ideological_position_of_A"],
                    "ideological_position_of_B": p["ideological_position_of_B"],
                    "response_condition": args.condition,
                    "raw_prompt": p["raw_prompt"],
                    "raw_output": mock_out,
                    "extracted_rationale": rationale,
                    "extracted_final_answer": final_ans,
                    "parser_status": status,
                    "answer_alignment": final_ans == p["sycophantic_aligned_answer"],
                    "run_id": "mock_run",
                    "decoding_config": {
                        "do_sample": cfg["do_sample"],
                        "temperature": cfg["temperature"],
                        "max_new_tokens": cfg["max_new_tokens"]
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Wrote mock results to {out_filename}")
        return

    # Load model and tokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16,
        device_map=cfg["device_map"]
    )
    model.eval()

    # Pre-tokenize to get exact token lengths for length-bucket sorting
    print("Pre-tokenizing inputs for length-bucket batching...")
    for item in prompts_to_run:
        messages = []
        if cfg["prompt_templates"].get("system"):
            messages.append({"role": "system", "content": cfg["prompt_templates"]["system"]})
        messages.append({"role": "user", "content": item["raw_prompt"]})
        
        try:
            chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            # Fallback if system role not supported (e.g. Gemma)
            if "system" in str(e).lower() or "role" in str(e).lower():
                user_content = f"{cfg['prompt_templates']['system']}\n\n{item['raw_prompt']}"
                messages = [{"role": "user", "content": user_content}]
                chat_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                raise e
                
        item["chat_prompt"] = chat_str
        item["length"] = len(tokenizer.encode(chat_str, add_special_tokens=False))


    # Sort by length
    prompts_to_run.sort(key=lambda x: x["length"])
    
    # Run batches
    batch_size = cfg["batch_size"]
    run_id = get_hash()
    
    with open(out_filename, "a", encoding="utf-8") as f:
        for idx in range(0, len(prompts_to_run), batch_size):
            batch = prompts_to_run[idx : idx + batch_size]
            prompts_batch = [b["chat_prompt"] for b in batch]
            
            inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True).to(model.device)
            
            t0 = time.time()
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=cfg["max_new_tokens"],
                        do_sample=cfg["do_sample"],
                        pad_token_id=tokenizer.pad_token_id
                    )
            except torch.cuda.OutOfMemoryError:
                print("CUDA OutOfMemoryError detected. Backing off to batch size 1 for this chunk...")
                torch.cuda.empty_cache()
                # Run elements in this batch one by one
                for b in batch:
                    single_inputs = tokenizer([b["chat_prompt"]], return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        outputs = model.generate(
                            **single_inputs,
                            max_new_tokens=cfg["max_new_tokens"],
                            do_sample=cfg["do_sample"],
                            pad_token_id=tokenizer.pad_token_id
                        )
                    raw_out = tokenizer.decode(outputs[0, single_inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    rationale, final_ans, status = extract_rationale_and_answer(raw_out, model_family)
                    
                    result = {
                        "model_id": model_id,
                        "model_family": model_family,
                        "model_revision": cfg.get("revision", "main"),
                        "question_id": b["question_id"],
                        "persona_condition": b["persona_condition"],
                        "persona_supported_option": b["sycophantic_aligned_answer"],
                        "ideological_position_of_A": b["ideological_position_of_A"],
                        "ideological_position_of_B": b["ideological_position_of_B"],
                        "response_condition": args.condition,
                        "raw_prompt": b["raw_prompt"],
                        "raw_output": raw_out,
                        "extracted_rationale": rationale,
                        "extracted_final_answer": final_ans,
                        "parser_status": status,
                        "answer_alignment": final_ans == b["sycophantic_aligned_answer"],
                        "run_id": run_id,
                        "decoding_config": {
                            "do_sample": cfg["do_sample"],
                            "temperature": cfg["temperature"],
                            "max_new_tokens": cfg["max_new_tokens"]
                        },
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                continue

            for batch_i, b in enumerate(batch):
                prompt_len = inputs.input_ids.shape[1]
                gen_tokens = outputs[batch_i, prompt_len:]
                raw_out = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                rationale, final_ans, status = extract_rationale_and_answer(raw_out, model_family)
                
                result = {
                    "model_id": model_id,
                    "model_family": model_family,
                    "model_revision": cfg.get("revision", "main"),
                    "question_id": b["question_id"],
                    "persona_condition": b["persona_condition"],
                    "persona_supported_option": b["sycophantic_aligned_answer"],
                    "ideological_position_of_A": b["ideological_position_of_A"],
                    "ideological_position_of_B": b["ideological_position_of_B"],
                    "response_condition": args.condition,
                    "raw_prompt": b["raw_prompt"],
                    "raw_output": raw_out,
                    "extracted_rationale": rationale,
                    "extracted_final_answer": final_ans,
                    "parser_status": status,
                    "answer_alignment": final_ans == b["sycophantic_aligned_answer"],
                    "run_id": run_id,
                    "decoding_config": {
                        "do_sample": cfg["do_sample"],
                        "temperature": cfg["temperature"],
                        "max_new_tokens": cfg["max_new_tokens"]
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"Batch {idx//batch_size + 1} complete in {time.time()-t0:.1f}s.")

    print(f"Success! Model outputs written to {out_filename}")

if __name__ == "__main__":
    main()
