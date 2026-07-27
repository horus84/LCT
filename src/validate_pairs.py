import json
import random

def main():
    with open("paired_dataset.json", "r", encoding="utf-8") as f:
        paired_dataset = json.load(f)

    print(f"Loaded {len(paired_dataset)} pairs.")
    
    with open("prompt_pairs_audit.md", "w", encoding="utf-8") as out:
        out.write("# Prompt Pairs Audit\n\n")
        out.write("This file shows 30 randomly selected paired prompts side-by-side to verify that ONLY the political identity phrases changed, while the name, background, question, and options remained identical.\n\n")
        
        sample = random.sample(paired_dataset, 30)
        
        for i, item in enumerate(sample):
            out.write(f"## Pair {i+1} (Original Alignment: {item['original_alignment'].upper()})\n")
            out.write("### Liberal Persona Prompt\n")
            out.write(f"```text\n{item['prompt_liberal']}\n```\n")
            out.write("### Conservative Persona Prompt\n")
            out.write(f"```text\n{item['prompt_conservative']}\n```\n")
            out.write("---\n\n")

    print("Created prompt_pairs_audit.md with 30 randomly selected pairs.")

if __name__ == "__main__":
    main()
