import os
import pandas as pd

def main():
    csv_path = "results/cross_model_results.csv"
    out_dir = "paper/tables"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/cross_model_table.tex"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Cannot generate LaTeX table.")
        return

    df = pd.read_csv(csv_path)

    # Format numbers as percentages
    df_tex = df.copy()
    for col in ["direct_accommodation_rate", "direct_anti_accommodation_rate", "direct_net_persona_effect",
                "cot_accommodation_rate", "cot_anti_accommodation_rate", "cot_net_persona_effect", "cot_minus_direct_diff"]:
        df_tex[col] = df_tex[col].apply(lambda x: f"{x*100:.1f}\%")

    df_tex["mcnemar_p"] = df_tex["mcnemar_p"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")

    # Generate LaTeX code
    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l ccc ccc c}",
        "\\toprule",
        " & \\multicolumn{3}{c}{\\textbf{Direct (Zero-Shot)}} & \\multicolumn{3}{c}{\\textbf{Chain-of-Thought (CoT)}} & \\textbf{CoT - Direct} \\\\",
        "\\cmidrule(r){2-4} \\cmidrule(l){5-7}",
        "\\textbf{Model Family} & \\textbf{N} & \\textbf{Acc\\%} & \\textbf{Net\\%} & \\textbf{N} & \\textbf{Acc\\%} & \\textbf{Net\\%} & \\textbf{Diff} ($p$-value) \\\\",
        "\\midrule"
    ]

    for _, row in df_tex.iterrows():
        line = (f"{row['model_family'].title()} & {row['direct_valid_N']} & {row['direct_accommodation_rate']} & "
                f"{row['direct_net_persona_effect']} & {row['cot_valid_N']} & {row['cot_accommodation_rate']} & "
                f"{row['cot_net_persona_effect']} & {row['cot_minus_direct_diff']} ({row['mcnemar_p']}) \\\\")
        latex_lines.append(line)

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Cross-Model Comparison of Persona Accommodation. Net\\% represents the Net Persona Effect (accommodation rate minus anti-accommodation rate). Diff indicates the CoT minus Direct difference with exact paired McNemar $p$-values.}",
        "\\label{tab:cross_model_comparison}",
        "\\end{table*}"
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")

    print(f"Successfully generated LaTeX cross-model table at {out_path}")

if __name__ == "__main__":
    main()
