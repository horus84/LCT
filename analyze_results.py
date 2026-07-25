import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import os

def main():
    if not os.path.exists('probe_results.csv'):
        print("probe_results.csv not found! Run bfcl_probe.py first.")
        return

    df = pd.read_csv('probe_results.csv')
    
    # Define semantic failure
    df['semantic_failure'] = ((df['fp'] == 1) | (df['fn'] == 1)).astype(int)
    
    configs = df['config'].unique()
    
    for config in configs:
        print(f"\n=============================================")
        print(f"Analysis for Configuration: {config}")
        print(f"=============================================")
        
        d_conf = df[df['config'] == config]
        
        # 1. Confusion Matrix & Mechanism counts
        tp = d_conf['tp'].sum()
        fp = d_conf['fp'].sum()
        fn = d_conf['fn'].sum()
        tn = d_conf['tn'].sum()
        mech_a = d_conf['mech_a'].sum()
        mech_b = d_conf['mech_b'].sum()
        
        print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"Mechanism A (Protocol Excluded): {mech_a}")
        print(f"Mechanism B (Semantic Failure given Protocol Reachable): {mech_b}")
        
        # 2. Mann-Whitney U test for projection tax
        # H0: tax for correct outputs == tax for incorrect outputs
        correct_tax = d_conf[d_conf['semantic_failure'] == 0]['cumulative_tax']
        incorrect_tax = d_conf[d_conf['semantic_failure'] == 1]['cumulative_tax']
        
        if len(correct_tax) > 0 and len(incorrect_tax) > 0:
            stat, p_val = mannwhitneyu(incorrect_tax, correct_tax, alternative='greater')
            print(f"Mann-Whitney U (Tax_incorrect > Tax_correct): p-value = {p_val:.4e}")
            if p_val < 0.05:
                print("  -> SIGNIFICANT difference: Incorrect outputs have higher projection tax.")
            else:
                print("  -> NO significant difference.")
        else:
            print("Mann-Whitney U: Not enough classes to compute.")
            
        # 3. ROC AUC
        if len(d_conf['semantic_failure'].unique()) > 1:
            auc = roc_auc_score(d_conf['semantic_failure'], d_conf['cumulative_tax'])
            print(f"ROC AUC (Tax predicting failure): {auc:.4f}")
            
            # Plot ROC
            fpr, tpr, _ = roc_curve(d_conf['semantic_failure'], d_conf['cumulative_tax'])
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: Cumulative Tax vs Failure ({config})')
            plt.legend(loc='lower right')
            plt.savefig(f'roc_{config}.png')
            plt.close()
        else:
            print("ROC AUC: Not computable (only one class present).")
            
        # 4. Histogram of Tax
        plt.figure(figsize=(8, 6))
        sns.histplot(data=d_conf, x='cumulative_tax', hue='semantic_failure', 
                     kde=True, bins=20, palette={0: 'blue', 1: 'red'})
        plt.title(f'Projection Tax Distribution ({config})')
        plt.xlabel('Cumulative Projection Tax')
        plt.savefig(f'tax_hist_{config}.png')
        plt.close()
        
        # 5. Scatter Plot: Decision Region Tax vs Argument Region Tax
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=d_conf, x='decision_region_tax', y='argument_region_tax', 
                        hue='semantic_failure', palette={0: 'blue', 1: 'red'})
        plt.title(f'Tax by Region ({config})')
        plt.savefig(f'scatter_{config}.png')
        plt.close()
        
    # Scientific Decision Logic
    print("\n=============================================")
    print("FINAL SCIENTIFIC DECISION")
    print("=============================================")
    d_a = df[df['config'] == 'A_COMPATIBLE']
    if d_a['mech_a'].sum() == d_a['tn'].sum() and d_a['mech_a'].sum() > 0:
        print("Verdict: INSTRUMENTATION BUG.")
        print("Mechanism A perfectly correlates with TN again. Definition of 'reachability' is flawed.")
    else:
        if len(d_a['semantic_failure'].unique()) > 1:
            auc = roc_auc_score(d_a['semantic_failure'], d_a['cumulative_tax'])
            if auc > 0.70:
                print("Verdict: PROMISING SIGNAL.")
                print("Projection Tax shows predictive power for semantic failure.")
            elif auc < 0.55:
                print("Verdict: NULL RESULT.")
                print("Projection Tax does not strongly predict semantic failure in Compatible configuration.")
            else:
                print("Verdict: INCONCLUSIVE.")
        else:
            print("Verdict: INCONCLUSIVE (Not enough data variance).")

if __name__ == '__main__':
    main()
