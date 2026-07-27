import os
import pandas as pd
import tempfile

def test_resume_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "results.csv")
        
        completed = set()
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                completed.add((row["sample_id"], float(row["truncation"])))
                
        assert len(completed) == 0
        
        mock_data = [
            {"sample_id": "q1_liberal", "truncation": 0.10, "predicted_answer": "A"},
            {"sample_id": "q1_liberal", "truncation": 0.50, "predicted_answer": "B"}
        ]
        df_mock = pd.DataFrame(mock_data)
        df_mock.to_csv(csv_path, index=False)
        
        completed2 = set()
        if os.path.exists(csv_path):
            df2 = pd.read_csv(csv_path)
            for _, row in df2.iterrows():
                completed2.add((row["sample_id"], float(row["truncation"])))
                
        assert len(completed2) == 2
        assert ("q1_liberal", 0.10) in completed2
        assert ("q1_liberal", 0.50) in completed2
        assert ("q1_liberal", 1.00) not in completed2
