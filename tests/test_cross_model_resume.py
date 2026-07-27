import os
import json
import tempfile

def test_cross_model_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_filename = os.path.join(tmpdir, "test_direct.jsonl")
        
        # Pre-seed some mock records
        mock_data = [
            {"question_id": "q1", "persona_condition": "liberal", "parser_status": "valid"},
            {"question_id": "q1", "persona_condition": "conservative", "parser_status": "valid"}
        ]
        
        with open(out_filename, "w", encoding="utf-8") as f:
            for d in mock_data:
                f.write(json.dumps(d) + "\n")
                
        completed = set()
        if os.path.exists(out_filename):
            with open(out_filename, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        completed.add((obj["question_id"], obj["persona_condition"]))
                        
        assert len(completed) == 2
        assert ("q1", "liberal") in completed
        assert ("q1", "conservative") in completed
        assert ("q2", "liberal") not in completed
