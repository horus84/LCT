import os
import json
import argparse
import yaml
import re

def audit_results(in_file, dataset_path, out_file):
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Input generation file not found: {in_file}. If running on Kaggle, please ensure the file is generated or placed in the working directory.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset_map = {item["question_id"]: item for item in dataset}

    total_records = 0
    cot_records = 0
    valid_cot_think = 0
    missing_think = 0
    unclosed_think = 0
    empty_think = 0

    suffix_counts = {}
    answer_formats = {}
    invalid_ids = []

    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            data = json.loads(line)
            q_id = data["question_id"]
            p_cond = data["persona_condition"]
            r_cond = data["response_condition"]
            response = data["response"]

            if r_cond != "cot":
                continue
            cot_records += 1

            has_start = "<think>" in response
            has_end = "</think>" in response

            if not has_start:
                missing_think += 1
                invalid_ids.append({"id": f"{q_id}_{p_cond}", "reason": "missing_think_tag"})
                continue
            
            if not has_end:
                unclosed_think += 1
                invalid_ids.append({"id": f"{q_id}_{p_cond}", "reason": "unclosed_think_tag"})
                continue

            think_part = response.split("</think>")[0].replace("<think>", "").strip()
            if not think_part:
                empty_think += 1
                invalid_ids.append({"id": f"{q_id}_{p_cond}", "reason": "empty_think_tag"})
                continue

            valid_cot_think += 1

            after_think = response.split("</think>")[-1]
            match = re.search(r'\b([AB])\b|\(([AB])\)', after_think.upper())
            if match:
                ans_str = match.group(0)
                start_idx = after_think.upper().find(ans_str)
                preceding_text = after_think[:start_idx]
                suffix_counts[preceding_text] = suffix_counts.get(preceding_text, 0) + 1
                answer_formats[ans_str] = answer_formats.get(ans_str, 0) + 1
            else:
                invalid_ids.append({"id": f"{q_id}_{p_cond}", "reason": "no_parsable_answer_after_think"})

    report = {
        "total_records": total_records,
        "cot_records": cot_records,
        "valid_cot_think": valid_cot_think,
        "missing_think": missing_think,
        "unclosed_think": unclosed_think,
        "empty_think": empty_think,
        "common_suffixes_before_answer": suffix_counts,
        "answer_formats": answer_formats,
        "exclusions": invalid_ids,
        "exclusion_count": len(invalid_ids)
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Stage 1 Audit Complete ===")
    print(f"Total CoT records: {cot_records}")
    print(f"Valid CoT with closed think: {valid_cot_think}")
    print(f"Exclusions: {len(invalid_ids)}")
    print(f"Audit report saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/early_answering.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    audit_results(cfg["in_file"], cfg["dataset_path"], cfg["audit_output"])
