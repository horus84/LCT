import os
import json
import argparse
from collections import defaultdict

def validate_file(in_file, report_dir):
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Input file not found: {in_file}")

    # Index by question_id
    question_map = defaultdict(list)
    
    total = 0
    valid_count = 0
    invalid_count = 0
    ambiguous_count = 0
    
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            total += 1
            data = json.loads(line)
            q_id = data["question_id"]
            p_cond = data["persona_condition"]
            status = data["parser_status"]
            
            if status == "valid":
                valid_count += 1
            elif status == "ambiguous":
                ambiguous_count += 1
            else:
                invalid_count += 1
                
            question_map[q_id].append(data)
            
    # Audit checks
    unique_qids = len(question_map)
    paired_complete = 0
    incomplete_qids = []
    
    for q_id, occurrences in question_map.items():
        # Check if we have both persona variants
        personas = [o["persona_condition"] for o in occurrences]
        if "liberal" in personas and "conservative" in personas:
            paired_complete += 1
        else:
            incomplete_qids.append(q_id)
            
    # Create report
    report = {
        "file_audited": in_file,
        "total_records": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "ambiguous_count": ambiguous_count,
        "unique_questions": unique_qids,
        "complete_pairs": paired_complete,
        "incomplete_questions": incomplete_qids,
        "status": "PASS" if (unique_qids == 100 and paired_complete == 100 and invalid_count == 0) else "WARNING"
    }
    
    os.makedirs(report_dir, exist_ok=True)
    basename = os.path.basename(in_file).replace(".jsonl", "_audit.json")
    report_path = f"{report_dir}/{basename}"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"=== Validation Report for {os.path.basename(in_file)} ===")
    print(f"  Total Records:    {total}")
    print(f"  Valid Parsed:     {valid_count} ({valid_count/total:.1%})")
    print(f"  Invalid / Amb:    {invalid_count} / {ambiguous_count}")
    print(f"  Unique Qs:        {unique_qids}")
    print(f"  Complete Pairs:   {paired_complete}")
    print(f"  Audit Status:     {report['status']}")
    print(f"  Audit Report written to {report_path}")
    print("==================================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to jsonl file to validate")
    parser.add_argument("--report_dir", type=str, default="results/model_audit_reports")
    args = parser.parse_args()
    
    validate_file(args.input, args.report_dir)

if __name__ == "__main__":
    main()
