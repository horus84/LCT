import json
import re

def swap_political_alignment(text, current_alignment):
    """
    Swaps the political alignment of the text.
    If current_alignment is 'liberal', swaps to 'conservative' equivalents.
    If current_alignment is 'conservative', swaps to 'liberal' equivalents.
    """
    if current_alignment == 'liberal':
        replacements = [
            (r'\bliberal\b', 'conservative', re.IGNORECASE),
            (r'\bprogressive\b', 'traditional', re.IGNORECASE),
            (r'\bdemocrat\b', 'republican', re.IGNORECASE),
            (r'\bleft-wing\b', 'right-wing', re.IGNORECASE),
            (r'\bfeminist\b', 'traditionalist', re.IGNORECASE),
            (r'\bDemocrats\b', 'Republicans', re.IGNORECASE),
            (r'universal healthcare, stronger gun control laws, and alternative energy subsidies', 
             'free markets, second amendment rights, and traditional values', re.IGNORECASE),
            (r'fighting for human rights, protecting the environment', 
             'protecting constitutional rights, promoting free enterprise', re.IGNORECASE)
        ]
    elif current_alignment == 'conservative':
        replacements = [
            (r'\bconservative\b', 'liberal', re.IGNORECASE),
            (r'\btraditional\b', 'progressive', re.IGNORECASE),
            (r'\brepublican\b', 'democrat', re.IGNORECASE),
            (r'\bright-wing\b', 'left-wing', re.IGNORECASE),
            (r'\btraditionalist\b', 'feminist', re.IGNORECASE),
            (r'\bRepublicans\b', 'Democrats', re.IGNORECASE),
            (r'free markets, second amendment rights, and traditional values', 
             'universal healthcare, stronger gun control laws, and alternative energy subsidies', re.IGNORECASE),
            (r'protecting constitutional rights, promoting free enterprise', 
             'fighting for human rights, protecting the environment', re.IGNORECASE)
        ]
    else:
        return text

    new_text = text
    for pattern, repl, flags in replacements:
        # Custom repl function to preserve case
        def replace_preserve_case(match):
            word = match.group(0)
            if word.istitle():
                return repl.title()
            elif word.isupper():
                return repl.upper()
            else:
                return repl.lower()
        new_text = re.sub(pattern, replace_preserve_case, new_text, flags=flags)
    return new_text

def classify_persona(prompt):
    p = prompt.lower()
    if 'liberal' in p or 'progressive' in p or 'democrat' in p:
        return 'liberal'
    elif 'conservative' in p or 'republican' in p or 'right-wing' in p or 'traditionalist' in p:
        return 'conservative'
    return 'unknown'

def main():
    with open("sycophancy_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    paired_dataset = []

    for item in data:
        q_id = item["id"]
        original_prompt = item["prompt"]
        original_alignment = classify_persona(original_prompt)
        
        if original_alignment == "unknown":
            continue

        swapped_prompt = swap_political_alignment(original_prompt, original_alignment)

        # Create Persona A (Liberal) and Persona B (Conservative)
        if original_alignment == "liberal":
            prompt_liberal = original_prompt
            prompt_conservative = swapped_prompt
            
            sycophantic_liberal = item["sycophantic_answer"]
            sycophantic_conservative = item["honest_answer"] # The honest answer for the liberal is the sycophantic for conservative
        else:
            prompt_conservative = original_prompt
            prompt_liberal = swapped_prompt
            
            sycophantic_conservative = item["sycophantic_answer"]
            sycophantic_liberal = item["honest_answer"]

        paired_dataset.append({
            "question_id": q_id,
            "prompt_liberal": prompt_liberal,
            "prompt_conservative": prompt_conservative,
            "sycophantic_liberal": sycophantic_liberal,
            "sycophantic_conservative": sycophantic_conservative,
            "original_alignment": original_alignment
        })

    with open("paired_dataset.json", "w", encoding="utf-8") as f:
        json.dump(paired_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Created paired dataset with {len(paired_dataset)} questions.")

if __name__ == "__main__":
    main()
