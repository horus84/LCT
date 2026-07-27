import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = "results/cross_model_statistics.json"
    out_dir = "paper/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/cross_model_comparison.png"

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Cannot generate comparison figure.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    models = list(stats.keys())
    
    # Values
    direct_rates = [stats[m]["direct_accommodation_rate"] * 100 for m in models]
    cot_rates = [stats[m]["cot_accommodation_rate"] * 100 for m in models]
    
    # Extract bootstrap CIs for error bars
    # bootstrap_ci_95 gives [low, high] for difference. For individual error bars, let's use standard error or mock standard CIs based on N
    # Let's calculate standard error of proportion: sqrt(p * (1-p) / N)
    direct_errs = []
    cot_errs = []
    
    for m in models:
        n_dir = stats[m]["direct_valid_n"]
        p_dir = stats[m]["direct_accommodation_rate"]
        direct_errs.append(1.96 * np.sqrt(p_dir * (1 - p_dir) / n_dir) * 100 if n_dir > 0 else 0)
        
        n_cot = stats[m]["cot_valid_n"]
        p_cot = stats[m]["cot_accommodation_rate"]
        cot_errs.append(1.96 * np.sqrt(p_cot * (1 - p_cot) / n_cot) * 100 if n_cot > 0 else 0)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    
    rects1 = ax.bar(x - width/2, direct_rates, width, yerr=direct_errs, label='Direct (Zero-Shot)', 
                    color='#3F51B5', capsize=5, alpha=0.9)
    rects2 = ax.bar(x + width/2, cot_rates, width, yerr=cot_errs, label='Chain-of-Thought (CoT)', 
                    color='#FF5722', capsize=5, alpha=0.9)

    ax.set_ylabel('Persona Accommodation Rate (%)')
    ax.set_title('Persona Accommodation Across Language Model Families')
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in models])
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Successfully generated comparison figure at {out_path}")

if __name__ == "__main__":
    main()
