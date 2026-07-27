from transformers import AutoTokenizer

def test_answer_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Test formats to check balance and ID distinctness
    formats = [
        ("A", "B"),
        (" A", " B"),
        ("(A", "(B")
    ]
    for a_str, b_str in formats:
        tokens_a = tokenizer.encode(a_str, add_special_tokens=False)
        tokens_b = tokenizer.encode(b_str, add_special_tokens=False)
        
        assert len(tokens_a) == len(tokens_b), f"Tokenization drift: {a_str} ({len(tokens_a)}) vs {b_str} ({len(tokens_b)})"
        assert tokens_a[0] != tokens_b[0], "Token IDs are identical"
