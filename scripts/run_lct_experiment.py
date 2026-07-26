import json
import argparse
import time
import os

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Warning: transformers/torch not found. Run 'pip install transformers torch accelerate'")
    exit(1)

def get_prompt(question, language):
    return f"""You are a helpful mathematical assistant.
Please solve the following problem.
You must think step-by-step. Put all your reasoning inside <think>...</think> tags.
CRITICAL CONSTRAINT: You must write your reasoning inside the <think> tags ENTIRELY in {language}. 
After the </think> tag, output your final answer in English.

Problem: {question}
"""

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Test-Time Scaling LCT Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    parser.add_argument("--samples", type=int, default=16, help="Number of Best-of-N samples per prompt")
    parser.add_argument("--out_file", type=str, default="lct_results.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    # 5 Harder reasoning questions for Phase 3
    questions = [
        "A baker has 3 bags of flour. Each bag contains 5 kilograms. He uses 2.5 kilograms to bake a batch of bread. If he bakes 4 batches, how much flour is left in grams?",
        "If x + y = 10 and x - y = 4, what is the value of x * y?",
        "A train leaves New York at 3:00 PM traveling at 60 mph. Another train leaves at 4:00 PM traveling at 80 mph on a parallel track. At what time will the second train catch up?",
        "What is the sum of the first 50 positive even integers?",
        "In a class of 30 students, 18 play soccer, 15 play basketball, and 5 play neither. How many students play both sports?"
    ]
    languages = ["English", "Spanish", "Hindi"]

    print(f"Loading {args.model} on device_map='auto' (Dual T4 recommended)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Use bfloat16 and auto device map to split across Dual T4s safely
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        print("Ensure you have enough VRAM and 'accelerate' installed.")
        exit(1)

    print(f"Model loaded successfully. Starting Best-of-{args.samples} sampling...")

    # Open file in append mode so we don't lose data if it crashes
    with open(args.out_file, "a", encoding="utf-8") as f:
        for q_idx, q in enumerate(questions):
            for lang in languages:
                print(f"\n--- Question {q_idx+1}/{len(questions)} | Language: {lang} ---")
                prompt = get_prompt(q, lang)
                
                messages = [
                    {"role": "system", "content": "You are a highly capable reasoning assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                for sample_idx in range(args.samples):
                    print(f"  Generating sample {sample_idx+1}/{args.samples}...")
                    start_t = time.time()
                    
                    try:
                        # Temperature 0.7 for diverse sampling in Best-of-N
                        outputs = model.generate(
                            **inputs, 
                            max_new_tokens=1024, 
                            do_sample=True, 
                            temperature=0.7,
                            top_p=0.9
                        )
                        gen_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                        latency = time.time() - start_t
                        
                        result_obj = {
                            "question": q,
                            "language": lang,
                            "sample_idx": sample_idx,
                            "response": gen_text,
                            "latency_sec": latency
                        }
                        
                        f.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                        f.flush()  # Force write to disk immediately
                        
                    except Exception as e:
                        print(f"  ERROR during generation: {e}")
                        # Continue to next sample rather than crashing the whole run
                        continue
                        
    print("\nExperiment complete. All results safely saved to", args.out_file)

if __name__ == "__main__":
    main()
