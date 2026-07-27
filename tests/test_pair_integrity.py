import json
import os
import re
import pytest

def test_pair_integrity():
    dataset_path = "paired_dataset.json"
    if not os.path.exists(dataset_path):
        pytest.skip("paired_dataset.json not found, skipping pair integrity test")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    assert len(dataset) > 0
    
    for item in dataset:
        p_lib = item["prompt_liberal"]
        p_con = item["prompt_conservative"]
        
        # Check similarity
        lib_words = set(re.findall(r'\w+', p_lib.lower()))
        con_words = set(re.findall(r'\w+', p_con.lower()))
        
        common = lib_words.intersection(con_words)
        total_union = lib_words.union(con_words)
        
        jaccard = len(common) / len(total_union)
        assert jaccard > 0.8, f"Paired prompts differ too much for {item['question_id']}: Jaccard={jaccard:.2f}"
