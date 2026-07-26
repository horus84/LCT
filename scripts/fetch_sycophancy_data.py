import urllib.request
import json
import random

def fetch_anthropic_sycophancy():
    url = "https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl"
    print(f"Downloading REAL Anthropic sycophancy dataset from {url}...")
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    dataset = []
    # Set seed for reproducibility
    random.seed(42)
    # Sample 100 questions for the Kaggle run to balance significance and compute
    sampled_lines = random.sample(lines, min(100, len(lines)))
    
    for i, line in enumerate(sampled_lines):
        obj = json.loads(line)
        # Anthropic formats the answer fields with a leading space like " (A)"
        dataset.append({
            "id": f"anthropic_sycophancy_{i}",
            "prompt": obj["question"].strip(),
            "sycophantic_answer": obj["answer_matching_behavior"].strip(),
            "honest_answer": obj["answer_not_matching_behavior"].strip()
        })
        
    out_file = "sycophancy_dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Successfully saved {len(dataset)} real sycophancy questions to {out_file}.")

if __name__ == "__main__":
    fetch_anthropic_sycophancy()
