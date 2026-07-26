import json
import argparse
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Warning: transformers/torch not found, script will run in dry-run mode.")
    torch = None

def get_prompt(question, language):
    return f"""You are a helpful mathematical assistant.
Please solve the following problem.
You must think step-by-step. Put all your reasoning inside <think>...</think> tags.
CRITICAL CONSTRAINT: You must write your reasoning inside the <think> tags ENTIRELY in {language}. 
After the </think> tag, output your final answer in English.

Problem: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use (defaulting to 7B for actual pilot)")
    parser.add_argument("--dry_run", action="store_true", help="Run without loading model")
    args = parser.parse_args()

    questions = [
        "A train travels 60 miles per hour. How long does it take to travel 150 miles?",
        "If 3x + 5 = 14, what is x?"
    ]
    languages = ["English", "Spanish", "Hindi"]

    results = []

    if not args.dry_run and torch is not None:
        print(f"Loading {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
    else:
        print("Dry run mode (or missing torch). Skipping actual model load.")

    for q in questions:
        for lang in languages:
            print(f"\n--- Testing Language: {lang} ---")
            prompt = get_prompt(q, lang)
            
            if not args.dry_run and torch is not None:
                messages = [
                    {"role": "system", "content": "You are a helpful reasoning assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                start_t = time.time()
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                gen_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                latency = time.time() - start_t
            else:
                # Mock generation
                gen_text = f"<think>\nMock {lang} reasoning steps...\n</think>\nFinal answer: mock."
                latency = 0.5
                time.sleep(0.1)
                
            print(f"Result:\n{gen_text}")
            
            results.append({
                "question": q,
                "language": lang,
                "response": gen_text,
                "latency_sec": latency
            })

    with open("pilot_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nPilot complete. Results saved to pilot_results.json.")

if __name__ == "__main__":
    main()
