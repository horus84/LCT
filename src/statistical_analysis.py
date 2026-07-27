import pandas as pd
import numpy as np
import json
import argparse
import sys
from scipy.stats import binomtest

def bootstrap_ci(data1, data2, n_boot=10000, alpha=0.05):
    """Compute paired bootstrap CI for the difference of means."""
    diffs = np.array(data2) - np.array(data1)
    boot_diffs = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot_diffs, [alpha/2 * 100, (1 - alpha/2) * 100])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="parsed_paired_results.csv")
    parser.add_argument("--ea_file", type=str, default="paired_early_answering.jsonl")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"File {args.csv_file} not found. Please run the generation and parsing scripts first.")
        sys.exit(0)

    valid_df = df[df['status'] == 'valid']
    
    # Restructure for paired analysis
    pivot_df = valid_df.pivot(index=['question_id', 'response_condition'], 
                              columns='persona_condition', 
                              values=['final_answer', 'is_aligned']).reset_index()
    pivot_df.columns = ['_'.join(col).strip('_') for col in pivot_df.columns.values]
    pivot_df = pivot_df.dropna(subset=['final_answer_liberal', 'final_answer_conservative'])
    
    # Mark accommodation status for each question
    pivot_df['perfect_accommodation'] = pivot_df['is_aligned_liberal'] & pivot_df['is_aligned_conservative']
    pivot_df['anti_accommodation'] = (~pivot_df['is_aligned_liberal']) & (~pivot_df['is_aligned_conservative'])
    pivot_df['switched'] = pivot_df['final_answer_liberal'] != pivot_df['final_answer_conservative']
    
    for cond in ["zero_shot", "cot"]:
        cond_df = pivot_df[pivot_df['response_condition'] == cond]
        total = len(cond_df)
        
        # 1. Ideology-normalized transition matrices
        print(f"=== 1. Ideology-Normalized Transition Matrix ({cond.upper()}) ===")
        # We classify answers as 'Aligned' (A) or 'Non-Aligned' (N) for that persona
        lib_aligned = cond_df['is_aligned_liberal']
        con_aligned = cond_df['is_aligned_conservative']
        matrix = pd.crosstab(lib_aligned, con_aligned, 
                             rownames=['Liberal Persona Aligned'], 
                             colnames=['Conservative Persona Aligned'])
        print(matrix)
        
        # 2 & 3. Counts of switches and binomial tests
        print(f"\n=== 2 & 3. Switch Directions ({cond.upper()}) ===")
        switches_to_aligned = len(cond_df[cond_df['switched'] & cond_df['perfect_accommodation']])
        switches_to_anti = len(cond_df[cond_df['switched'] & cond_df['anti_accommodation']])
        total_switches = switches_to_aligned + switches_to_anti
        
        print(f"Total pairs that switched answer based on persona: {total_switches}")
        print(f"  Switches TOWARD persona (Persona-following): {switches_to_aligned}")
        print(f"  Switches AWAY from persona (Anti-accommodation): {switches_to_anti}")
        
        if total_switches > 0:
            b_test = binomtest(switches_to_aligned, total_switches, 0.5)
            print(f"Exact Directional Binomial Test (Towards vs Away): p = {b_test.pvalue:.4f}")
        print("\n")

    # 4, 5, 6. ZS vs CoT Accommodation Comparison
    print("=== 4. ZS vs CoT Accommodation Transition Matrix ===")
    zs_df = pivot_df[pivot_df['response_condition'] == 'zero_shot'][['question_id', 'perfect_accommodation']]
    cot_df = pivot_df[pivot_df['response_condition'] == 'cot'][['question_id', 'perfect_accommodation']]
    merged = pd.merge(zs_df, cot_df, on='question_id', suffixes=('_zs', '_cot')).dropna()
    
    zs_cot_matrix = pd.crosstab(merged['perfect_accommodation_zs'], merged['perfect_accommodation_cot'],
                                rownames=['Zero-Shot Accommodation'], colnames=['CoT Accommodation'])
    print(zs_cot_matrix)
    
    b = zs_cot_matrix.loc[True, False] if (True in zs_cot_matrix.index and False in zs_cot_matrix.columns) else 0
    c = zs_cot_matrix.loc[False, True] if (False in zs_cot_matrix.index and True in zs_cot_matrix.columns) else 0
    mcnemar_zs_cot = binomtest(c, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    print(f"\n=== 5. Exact Paired McNemar Test (ZS vs CoT) ===")
    print(f"Switched from ZS-Acc to CoT-NonAcc: {b}")
    print(f"Switched from ZS-NonAcc to CoT-Acc: {c}")
    print(f"McNemar p-value: {mcnemar_zs_cot:.4f}")
    
    zs_acc_rates = merged['perfect_accommodation_zs'].astype(int).tolist()
    cot_acc_rates = merged['perfect_accommodation_cot'].astype(int).tolist()
    ci_low, ci_high = bootstrap_ci(zs_acc_rates, cot_acc_rates)
    diff_mean = np.mean(cot_acc_rates) - np.mean(zs_acc_rates)
    print(f"\n=== 6. Paired Bootstrap CI (CoT - ZS) ===")
    print(f"Difference: {diff_mean*100:.1f}%  | 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]\n")

    # Early Answering Analysis
    try:
        ea_data = []
        with open(args.ea_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): ea_data.append(json.loads(line))
                
        if not ea_data: raise FileNotFoundError
        
        # Merge EA with original generation for 100% check
        ea_df = pd.DataFrame(ea_data)
        
        # We need the final answer from CoT to check full-answer agreement
        cot_results = valid_df[valid_df['response_condition'] == 'cot']
        
        print("=== 7 & 8 & 10. Early Answering Trajectories & Verification ===")
        truncs = ["0.10", "0.50", "0.75", "1.00"]
        print(f"{'Trunc':<6} | {'Denom':<6} | {'Alignment%':<12} | {'Matches Full CoT%'}")
        
        for t in truncs:
            aligned_count = 0
            matches_full_count = 0
            denom = 0
            
            for item in ea_data:
                trunc_data = item.get("truncations", {}).get(t)
                if not trunc_data: continue
                denom += 1
                
                gen_tok = trunc_data["generated_token"]
                
                if trunc_data["is_aligned"]:
                    aligned_count += 1
                    
                # 10. Check if it reproduces original final answer
                # Find original answer
                orig_match = cot_results[(cot_results['question_id'] == item['question_id']) & 
                                         (cot_results['persona_condition'] == item['persona_condition'])]
                if not orig_match.empty:
                    orig_ans = orig_match.iloc[0]['final_answer']
                    if gen_tok == orig_ans:
                        matches_full_count += 1
                        
            print(f"{float(t)*100:>4.0f}% | {denom:<6} | {aligned_count/denom if denom else 0:.1%}       | {matches_full_count/denom if denom else 0:.1%}")
            
        print("\n=== 9. Conditional Early-Answer Trajectories (Alignment Rate) ===")
        # Get question sets from CoT pivot
        cot_pivot = pivot_df[pivot_df['response_condition'] == 'cot']
        
        # a. all pairs (already calculated above, but per-question means looking at both conditions)
        # b. persona-sensitive pairs (switched)
        sensitive_qids = set(cot_pivot[cot_pivot['switched'] & cot_pivot['perfect_accommodation']]['question_id'])
        # c. perfect-accommodation pairs (same as above conceptually, or just any perfect accommodation)
        perfect_qids = set(cot_pivot[cot_pivot['perfect_accommodation']]['question_id'])
        # d. fixed-answer pairs (!switched)
        fixed_qids = set(cot_pivot[~cot_pivot['switched']]['question_id'])
        
        for name, qid_set in [("a. All Pairs", set(cot_pivot['question_id'])),
                              ("b. Persona-Sensitive (Switched to Align)", sensitive_qids),
                              ("c. Perfect Accommodation", perfect_qids),
                              ("d. Fixed-Answer Pairs", fixed_qids)]:
            print(f"--- {name} (N={len(qid_set)} questions) ---")
            for t in truncs:
                subset = [i for i in ea_data if i['question_id'] in qid_set]
                if not subset: continue
                aligned = sum(1 for i in subset if i["truncations"].get(t, {}).get("is_aligned"))
                total = sum(1 for i in subset if t in i["truncations"])
                print(f"  {float(t)*100:>4.0f}%: {aligned/total if total else 0:.1%} (denom={total})")
                
    except FileNotFoundError:
        print("Early answering file not found.")

if __name__ == "__main__":
    main()
