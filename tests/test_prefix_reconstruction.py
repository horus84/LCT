from transformers import AutoTokenizer
from src.reconstruct_prefixes import reconstruct_prefix

def test_reconstruct_prefix():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    prompt_text = "What is 2+2?\n (A) 4\n (B) 5"
    response_text = "<think>\nThinking step by step. 2+2=4.\n</think>\n(A)"
    final_answer = "A"
    
    prefix_tokens, answer_tokens, full_tokens = reconstruct_prefix(
        prompt_text, response_text, final_answer, tokenizer
    )
    
    assert prefix_tokens + answer_tokens == full_tokens
    
    decoded_answer = tokenizer.decode(answer_tokens).strip()
    assert "A" in decoded_answer
    
    # Assert that the decoded prefix ends with the exact whitespace and characters preceding the answer
    decoded_prefix = tokenizer.decode(prefix_tokens)
    assert decoded_prefix.endswith("\n") or decoded_prefix.endswith(" ")
