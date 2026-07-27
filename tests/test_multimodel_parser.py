import pytest
from src.run_multimodel_generation import extract_rationale_and_answer, clean_answer

def test_clean_answer():
    # Test cases
    assert clean_answer("Final answer: A") == ("A", "valid")
    assert clean_answer("Final answer: B") == ("B", "valid")
    assert clean_answer("Final answer: (A)") == ("A", "valid")
    assert clean_answer("Option (B)") == ("B", "valid")
    assert clean_answer("The answer is A.") == ("A", "valid")
    assert clean_answer("A or B is the answer") == (None, "ambiguous")
    assert clean_answer("Answer is C") == (None, "invalid")
    
def test_extract_rationale_and_answer():
    # Test CoT format with think tags
    response_think = "<think>\nReasoning here.\n</think>\nFinal answer: A"
    rationale, ans, status = extract_rationale_and_answer(response_think, "qwen")
    assert rationale == "Reasoning here."
    assert ans == "A"
    assert status == "valid"
    
    # Test CoT format without think tags but with Final answer:
    response_no_tags = "Reasoning step 1. Reasoning step 2. Final answer: B"
    rationale, ans, status = extract_rationale_and_answer(response_no_tags, "llama")
    assert rationale == "Reasoning step 1. Reasoning step 2."
    assert ans == "B"
    assert status == "valid"
