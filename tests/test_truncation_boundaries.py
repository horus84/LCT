from transformers import AutoTokenizer
import re

def extract_think_block(response_text):
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def test_truncation_boundaries():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    response_text = "<think>First token. Second token. Third token. Fourth token.</think>\n(A)"
    
    think_block = extract_think_block(response_text)
    assert think_block == "First token. Second token. Third token. Fourth token."
    
    think_tokens = tokenizer.encode(think_block, add_special_tokens=False)
    total_tokens = len(think_tokens)
    
    for frac in [0.10, 0.50, 0.75, 1.0]:
        n_tokens = max(1, int(total_tokens * frac))
        trunc_tokens = think_tokens[:n_tokens]
        assert len(trunc_tokens) == n_tokens
        assert n_tokens <= total_tokens
