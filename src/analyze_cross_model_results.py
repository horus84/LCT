import os
import json
import argparse
import pandas as pd
import numpy as np
from scipy.stats import binomtest

def bootstrap_ci(data1, data2, n_boot=10000, alpha=0.05):
    diffs = np.array(data2) - np.array(data1)
    boot_diffs = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot_diffs, [alpha/2 * 100, (1 - alpha/2) * 100])

def analyze_model(direct_file, cot_file):
    # Helper to load and pivot files
    def load_and_pivot(file_path):
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        df = df[df["parser_status"] == "valid"]
        pivot = df.pivot(index="question_id", columns="persona_condition", 
                          values=["extracted_final_answer", "answer_alignment", "persona_supported_option"]).reset_index()
        pivot.columns = ['_'.join(col).strip('_') for col in pivot.columns.values]
        pivot = pivot.dropna(subset=["extracted_final_answer_liberal", "extracted_final_answer_conservative"])
        return pivot

    print(f"\nAnalyzing replication files:\n  Direct: {direct_file}\n  CoT:    {cot_file}")
    
    df_dir = load_and_pivot(direct_file)
    df_cot = load_and_pivot(cot_file)
    
    # Calculate direct statistics
    dir_n = len(df_dir)
    df_dir["perfect_acc"] = df_dir["answer_alignment_liberal"] & df_dir["answer_alignment_conservative"]
    df_dir["anti_acc"] = (~df_dir["answer_alignment_liberal"]) & (~df_dir["answer_alignment_conservative"])
    df_dir["switched"] = df_dir["extracted_final_answer_liberal"] != df_dir["extracted_final_answer_conservative"]
    
    dir_acc_rate = df_dir["perfect_acc"].mean()
    dir_anti_rate = df_dir["anti_acc"].mean()
    dir_net_effect = dir_acc_rate - dir_anti_rate
    
    # Calculate CoT statistics
    cot_n = len(df_cot)
    df_cot["perfect_acc"] = df_cot["answer_alignment_liberal"] & df_cot["answer_alignment_conservative"]
    df_cot["anti_acc"] = (~df_cot["answer_alignment_liberal"]) & (~df_cot["answer_alignment_conservative"])
    df_cot["switched"] = df_cot["extracted_final_answer_liberal"] != df_cot["extracted_final_answer_conservative"]
    
    cot_acc_rate = df_cot["perfect_acc"].mean()
    cot_anti_rate = df_cot["anti_acc"].mean()
    cot_net_effect = cot_acc_rate - cot_anti_rate

    # Switched tests (Direct)
    dir_sw_to_aligned = len(df_dir[df_dir["switched"] & df_dir["perfect_acc"]])
    dir_sw_to_anti = len(df_dir[df_dir["switched"] & df_dir["anti_acc"]])
    dir_total_sw = dir_sw_to_aligned + dir_sw_to_anti
    dir_binom_p = binomtest(dir_sw_to_aligned, dir_total_sw, 0.5).pvalue if dir_total_sw > 0 else 1.0

    # Switched tests (CoT)
    cot_sw_to_aligned = len(df_cot[df_cot["switched"] & df_cot["perfect_acc"]])
    cot_sw_to_anti = len(df_cot[df_cot["switched"] & df_cot["anti_acc"]])
    cot_total_sw = cot_sw_to_aligned + cot_sw_to_anti
    cot_binom_p = binomtest(cot_sw_to_aligned, cot_total_sw, 0.5).pvalue if cot_total_sw > 0 else 1.0

    # Direct vs CoT comparison
    merged = pd.merge(df_dir[["question_id", "perfect_acc"]], 
                      df_cot[["question_id", "perfect_acc"]], 
                      on="question_id", suffixes=("_dir", "_cot"))
    
    m_matrix = pd.crosstab(merged["perfect_acc_dir"], merged["perfect_acc_cot"])
    b = m_matrix.loc[True, False] if (True in m_matrix.index and False in m_matrix.columns) else 0
    c = m_matrix.loc[False, True] if (False in m_matrix.index and True in m_matrix.columns) else 0
    
    mcnemar_p = binomtest(c, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    
    ci_low, ci_high = bootstrap_ci(merged["perfect_acc_dir"].astype(int).tolist(), 
                                   merged["perfect_acc_cot"].astype(int).tolist())
    
    diff_mean = cot_acc_rate - dir_acc_rate

    stats = {
        "direct_valid_n": dir_n,
        "direct_accommodation_rate": dir_acc_rate,
        "direct_anti_accommodation_rate": dir_anti_rate,
        "direct_net_persona_effect": dir_net_effect,
        "direct_switches": dir_total_sw,
        "direct_binom_p": dir_binom_p,
        "cot_valid_n": cot_n,
        "cot_accommodation_rate": cot_acc_rate,
        "cot_anti_accommodation_rate": cot_anti_rate,
        "cot_net_persona_effect": cot_net_effect,
        "cot_switches": cot_total_sw,
        "cot_binom_p": cot_binom_p,
        "difference_cot_minus_direct": diff_mean,
        "mcnemar_p": mcnemar_p,
        "bootstrap_ci_95": [ci_low, ci_high]
    }
    
    print(f"  Direct Acc Rate: {dir_acc_rate:.1%}  | Net Effect: {dir_net_effect*100:.1f}pp")
    print(f"  CoT Acc Rate:    {cot_acc_rate:.1%}  | Net Effect: {cot_net_effect*100:.1f}pp")
    print(f"  Difference:      {diff_mean*100:+.1f}pp (McNemar p = {mcnemar_p:.4f})")
    print(f"  Bootstrap CI:    [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, 
                        help="List of model results files: direct and CoT pairs. E.g. results/llama_direct.jsonl results/llama_cot.jsonl")
    parser.add_argument("--output_csv", type=str, default="results/cross_model_results.csv")
    parser.add_argument("--output_json", type=str, default="results/cross_model_statistics.json")
    args = parser.parse_args()

    # Match direct and CoT files by model family
    files_by_family = {}
    for f_path in args.inputs:
        basename = os.path.basename(f_path)
        if "_direct.jsonl" in basename:
            family = basename.split("_direct.jsonl")[0]
            if family not in files_by_family: files_by_family[family] = {}
            files_by_family[family]["direct"] = f_path
        elif "_cot.jsonl" in basename:
            family = basename.split("_cot.jsonl")[0]
            if family not in files_by_family: files_by_family[family] = {}
            files_by_family[family]["cot"] = f_path

    results_table = []
    full_stats = {}

    for family, paths in files_by_family.items():
        if "direct" in paths and "cot" in paths:
            stats = analyze_model(paths["direct"], paths["cot"])
            full_stats[family] = stats
            
            results_table.append({
                "model_family": family,
                "direct_valid_N": stats["direct_valid_n"],
                "direct_accommodation_rate": stats["direct_accommodation_rate"],
                "direct_anti_accommodation_rate": stats["direct_anti_accommodation_rate"],
                "direct_net_persona_effect": stats["direct_net_persona_effect"],
                "cot_valid_N": stats["cot_valid_n"],
                "cot_accommodation_rate": stats["cot_accommodation_rate"],
                "cot_anti_accommodation_rate": stats["cot_anti_accommodation_rate"],
                "cot_net_persona_effect": stats["cot_net_persona_effect"],
                "cot_minus_direct_diff": stats["difference_cot_minus_direct"],
                "mcnemar_p": stats["mcnemar_p"]
            })

    # Save outputs
    df_table = pd.DataFrame(results_table)
    df_table.to_csv(args.output_csv, index=False)
    
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2)

    print(f"\nWritten cross-model summary CSV to {args.output_csv}")
    print(f"Written cross-model detailed JSON statistics to {args.output_json}")

if __name__ == "__main__":
    main()
