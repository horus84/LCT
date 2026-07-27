import pandas as pd
import json
import argparse
import sys

def mcnemar_test(b, c):
    """Exact McNemar test for paired nominal data."""
    from scipy.stats import binomtest
    if b + c == 0:
        return 1.0
    return binomtest(b, b + c, 0.5).pvalue

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

    print("=== Table 1: Generation Counts ===")
    print(f"Total rows: {len(df)}")
    valid_df = df[df['status'] == 'valid']
    print(f"Valid outputs: {len(valid_df)}")
    print(f"Invalid outputs: {len(df) - len(valid_df)}")
    print("\n")

    # Pivot to create paired outcomes
    # We want rows: question_id, response_condition
    # Columns: final_answer under liberal, final_answer under conservative
    pivot_df = valid_df.pivot(index=['question_id', 'response_condition'], 
                              columns='persona_condition', 
                              values='final_answer').reset_index()
                              
    pivot_df = pivot_df.dropna() # Drop questions where one persona failed to generate a valid answer

    for cond in ["zero_shot", "cot"]:
        print(f"=== Table 2: Transition Matrix ({cond.upper()}) ===")
        cond_df = pivot_df[pivot_df['response_condition'] == cond]
        
        # Cross-tabulate Lib vs Con answers
        matrix = pd.crosstab(cond_df['liberal'], cond_df['conservative'], rownames=['Liberal Persona'], colnames=['Conservative Persona'])
        print(matrix)
        
        # McNemar Test: 
        # b = Number of times Liberal persona caused answer 'A' and Conservative persona caused answer 'B'
        # c = Number of times Liberal persona caused answer 'B' and Conservative persona caused answer 'A'
        if 'A' in matrix.index and 'B' in matrix.columns:
            b = matrix.loc['A', 'B'] if 'A' in matrix.index and 'B' in matrix.columns else 0
            c = matrix.loc['B', 'A'] if 'B' in matrix.index and 'A' in matrix.columns else 0
            
            p_val = mcnemar_test(b, c)
            print(f"\nMcNemar Test for paired switches (A->B vs B->A): p = {p_val:.4f}")
            
            # Persona Accommodation Rate
            # How often did the model switch its answer to ALIGN with the persona?
            # We need to look at 'is_aligned' rather than raw A/B.
        print("\n")

    print("=== Table 3: Persona Alignment Results ===")
    # Re-pivot on alignment
    align_df = valid_df.pivot(index=['question_id', 'response_condition'], 
                              columns='persona_condition', 
                              values='is_aligned').reset_index()
    align_df = align_df.dropna()

    for cond in ["zero_shot", "cot"]:
        c_df = align_df[align_df['response_condition'] == cond]
        lib_align = c_df['liberal'].mean()
        con_align = c_df['conservative'].mean()
        
        # Did the model give the persona-aligned answer in BOTH cases?
        # True sycophancy (accommodation) means switching answer to match persona.
        # This corresponds to is_aligned == True for BOTH personas for the SAME question.
        switched_to_align = (c_df['liberal'] & c_df['conservative']).sum()
        total = len(c_df)
        
        print(f"Condition: {cond.upper()}")
        print(f"  Liberal Persona Alignment Rate:      {lib_align:.1%}")
        print(f"  Conservative Persona Alignment Rate: {con_align:.1%}")
        print(f"  Perfect Persona Accommodation Rate:  {switched_to_align}/{total} ({switched_to_align/total:.1%})")
        print("\n")

    print("=== Table 4: Early Answering Trajectories ===")
    try:
        ea_data = []
        with open(args.ea_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ea_data.append(json.loads(line))
                    
        if ea_data:
            truncs = ["0.10", "0.50", "0.75", "1.00"]
            print(f"{'Truncation':<12} | {'Alignment Rate'}")
            for t in truncs:
                aligned = sum(1 for item in ea_data if item["truncations"].get(t, {}).get("is_aligned"))
                total = len(ea_data)
                print(f"{float(t)*100:>10.0f}% | {aligned/total:.1%}")
    except FileNotFoundError:
        print(f"Early answering file {args.ea_file} not found.")

if __name__ == "__main__":
    main()
