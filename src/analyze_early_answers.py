import os
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/early_answer_scores.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: early answering score file {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    
    print("=== Early Answering Trajectory Metrics ===")
    truncs = sorted(df["truncation"].unique())
    
    # 1. Trajectories by all pairs
    print("\n--- a. All Pairs ---")
    for t in truncs:
        sub = df[df["truncation"] == t]
        denom = len(sub)
        matches = sub["matches_original"].sum()
        aligned = sub["persona_alignment"].sum()
        print(f"Truncation {t*100:>3.0f}% | Denom: {denom:<4} | Original Answer Agreement: {matches/denom:.1%} | Persona Alignment Rate: {aligned/denom:.1%}")

    # We also want to compute categories
    # Read the generation file or recreate transition category mappings
    # If the user has transition info or categories, we can map them
    # For now, let's output basic trajectories grouped by persona_condition
    for cond in ["liberal", "conservative"]:
        print(f"\n--- Condition: {cond.upper()} Persona ---")
        for t in truncs:
            sub = df[(df["truncation"] == t) & (df["persona_condition"] == cond)]
            denom = len(sub)
            if denom == 0: continue
            matches = sub["matches_original"].sum()
            aligned = sub["persona_alignment"].sum()
            print(f"  {t*100:>3.0f}% | Denom: {denom:<4} | Agreement: {matches/denom:.1%} | Alignment: {aligned/denom:.1%}")

if __name__ == "__main__":
    main()
