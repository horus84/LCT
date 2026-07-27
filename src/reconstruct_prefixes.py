import os
import json
import re

def get_cot_prompt(question):
    return f"""You are a careful, logical reasoning assistant (System 2).
Please answer the following user query. You MUST think step-by-step and place all your logical reasoning inside <think>...</think> tags. 
After closing the </think> tag, output ONLY the final option letter inside parentheses, e.g., (A) or (B).

User Query: {question}
Answer:"""

def reconstruct_prefix(prompt_text, response_text, final_answer, tokenizer):
    """
    Tokenizes the combined chat prompt and assistant response,
    then slices it right before the token corresponding to the final answer.
    """
    messages = [
        {"role": "system", "content": "You are a highly capable AI assistant."},
        {"role": "user", "content": get_cot_prompt(prompt_text)}
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_str + response_text

    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    clean_ans = final_answer.replace("(", "").replace(")", "").strip()
    
    split_idx = None
    matched_suffix = None
    
    # Search from the end of the tokens for the first token representation of the final answer
    for i in range(1, min(15, len(full_tokens)) + 1):
        suffix_tokens = full_tokens[-i:]
        suffix_str = tokenizer.decode(suffix_tokens).strip()
        
        # Check if the decoded suffix is a match for the target answer format
        if suffix_str in [clean_ans, f"({clean_ans})", f" {clean_ans}", f"({clean_ans}"]:
            split_idx = len(full_tokens) - i
            matched_suffix = suffix_tokens
            break

    if split_idx is None:
        # Fallback to text split if token search fails
        print(f"Warning: Token search failed for answer '{clean_ans}'. Using fallback text split.")
        # Find the last occurrence of the clean answer in the last 20 characters
        # (This is just a fallback, should rarely be hit if the tokenizer matches)
        pattern = re.compile(rf'\b{clean_ans}\b|\({clean_ans}\)', re.IGNORECASE)
        matches = list(pattern.finditer(response_text))
        if matches:
            last_match = matches[-1]
            split_char_idx = len(prompt_str) + last_match.start()
            prefix_text = full_text[:split_char_idx]
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            # Find closest token boundary
            split_idx = len(prefix_tokens)
            matched_suffix = full_tokens[split_idx:]
        else:
            raise ValueError(f"Could not locate answer boundary for answer: {clean_ans}")

    prefix_tokens = full_tokens[:split_idx]
    answer_tokens = full_tokens[split_idx:]
    
    return prefix_tokens, answer_tokens, full_tokens
