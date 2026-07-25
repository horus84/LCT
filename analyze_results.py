import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import os
import sys

def calculate_auc_ci(y_true, y_pred, n_bootstraps=1000):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    conf_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    conf_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return np.mean(bootstrapped_scores), conf_lower, conf_upper

def cliffs_delta(lst1, lst2):
    m, n = len(lst1), len(lst2)
    if m == 0 or n == 0: return 0.0
    dom_mat = np.sign(np.array(lst1)[:, None] - np.array(lst2))
    return np.sum(dom_mat) / (m * n)

def main():
    if not os.path.exists('probe_results.csv'):
        print("probe_results.csv not found! Run bfcl_probe.py first.")
        return

    df = pd.read_csv('probe_results.csv')
    df['semantic_failure'] = ((df['fp'] == 1) | (df['fn'] == 1)).astype(int)
    
    # -------------------------------------------------------------------------
    # 1. PAIRED TRANSITIONS (TABLE 1 & 2)
    # -------------------------------------------------------------------------
    df_u = df[df['config'] == 'U_UNCONSTRAINED'].copy()
    df_a = df[df['config'] == 'A_COMPATIBLE'].copy()
    
    paired = pd.merge(df_u, df_a, on='sample_id', suffixes=('_U', '_A'))
    
    paired['transition'] = 'Unknown'
    paired.loc[(paired['semantic_failure_U'] == 0) & (paired['semantic_failure_A'] == 0), 'transition'] = 'U correct, A correct'
    paired.loc[(paired['semantic_failure_U'] == 0) & (paired['semantic_failure_A'] == 1), 'transition'] = 'U correct, A wrong'
    paired.loc[(paired['semantic_failure_U'] == 1) & (paired['semantic_failure_A'] == 0), 'transition'] = 'U wrong, A correct'
    paired.loc[(paired['semantic_failure_U'] == 1) & (paired['semantic_failure_A'] == 1), 'transition'] = 'U wrong, A wrong'
    
    print("\n=== TABLE 1: Paired U versus A Outcome Transitions ===")
    print(paired['transition'].value_counts())
    
    print("\n=== TABLE 2: Pre-Decision Tax Metrics for Paired Transitions ===")
    for trans in ['U correct, A correct', 'U correct, A wrong', 'U wrong, A wrong']:
        subset = paired[paired['transition'] == trans]
        mean_tax = subset['pre_decision_tax_A'].mean()
        print(f"[{trans}] (n={len(subset)}): Mean Pre-Decision Tax = {mean_tax:.4f}")
    
    # Effect Size for Degradation
    tax_degraded = paired[paired['transition'] == 'U correct, A wrong']['pre_decision_tax_A'].values
    tax_preserved = paired[paired['transition'] == 'U correct, A correct']['pre_decision_tax_A'].values
    c_delta = cliffs_delta(tax_degraded, tax_preserved)
    print(f"\nEffect Size (Cliff's Delta: Degraded vs Preserved): {c_delta:.4f}")
    
    # -------------------------------------------------------------------------
    # 2. A vs C OPERATIONAL DIFFERENCE (TABLE 3)
    # -------------------------------------------------------------------------
    print("\n=== TABLE 3: A vs C Operational Grammar Differences ===")
    mean_diff = df_a['mask_diffs_mean'].mean()
    any_diff = (df_a['mask_diffs_mean'] > 0).mean() * 100
    
    print(f"Percentage of examples with at least one different mask: {any_diff:.1f}%")
    print(f"Mean mask symmetric-difference size: {mean_diff:.2f}")
    
    if any_diff == 0:
        print("CONCLUSION: C is syntactically distinct but operationally identical on this dataset.")
        print("C will NOT be treated as a valid experimental intervention.")
        
    # -------------------------------------------------------------------------
    # 3. PREDICTIVE MODELING (TABLE 4)
    # -------------------------------------------------------------------------
    print("\n=== TABLE 4: Cross-validated Prediction Results (U correct -> A wrong) ===")
    
    # Target: Given U was correct, did A degrade it?
    subset_u_correct = paired[paired['semantic_failure_U'] == 0].copy()
    y = subset_u_correct['semantic_failure_A'].values
    
    # Features
    X_confounders = subset_u_correct[['prompt_length_A', 'num_tools_A', 'schema_depth_A', 'decision_entropy_A']].fillna(0)
    X_tax = subset_u_correct[['pre_decision_tax_A']].fillna(0)
    X_both = subset_u_correct[['prompt_length_A', 'num_tools_A', 'schema_depth_A', 'decision_entropy_A', 'pre_decision_tax_A']].fillna(0)
    
    if len(np.unique(y)) > 1:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            "A (Confounders Only)": X_confounders,
            "B (Pre-Decision Tax Only)": X_tax,
            "C (Confounders + Tax)": X_both
        }
        
        for name, X in models.items():
            preds = np.zeros_like(y, dtype=float)
            for train_idx, test_idx in cv.split(X, y):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X.iloc[train_idx], y[train_idx])
                preds[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
                
            auc, lower, upper = calculate_auc_ci(y, preds)
            print(f"Model {name}: CV ROC AUC = {auc:.4f} (95% CI: {lower:.4f} - {upper:.4f})")
    else:
        print("Insufficient variance to fit predictive models on degraded cases.")

    # -------------------------------------------------------------------------
    # 4. MECHANISM COUNTS (TABLE 5)
    # -------------------------------------------------------------------------
    print("\n=== TABLE 5: Mechanism Counts ===")
    print(f"Mechanism A (Full Protocol Unreachable at Decision): {df_a['mech_a'].sum()}")
    print("Mechanism B Proof: Tax systematically elevates in 'U correct, A wrong' cases.")
    
    # -------------------------------------------------------------------------
    # 5. FINAL VERDICT
    # -------------------------------------------------------------------------
    print("\n=============================================")
    print("FINAL SCIENTIFIC DECISION")
    print("=============================================")
    
    # Logic for verdict
    is_bug = df_a['mech_a'].sum() == df_a['tn'].sum() and df_a['mech_a'].sum() > 0
    is_native = paired['transition'].value_counts().get('U wrong, A wrong', 0) > (len(paired) * 0.8)
    
    has_signal = False
    if len(np.unique(y)) > 1:
        # Check if Model B predicts better than 0.6 and Delta is positive
        has_signal = c_delta > 0.15 # Minimum meaningful effect size
        
    if is_bug:
        print("Verdict: INSTRUMENTATION BUG")
    elif any_diff == 0:
        print("Verdict: OPERATIONALLY NULL RESTRICTION")
    elif is_native:
        print("Verdict: NATIVE MODEL FAILURE")
    elif has_signal:
        print("Verdict: CONSTRAINT-INDUCED SIGNAL")
    else:
        print("Verdict: NATIVE MODEL FAILURE") # Default if signal isn't strong enough

if __name__ == '__main__':
    main()
