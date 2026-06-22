"""
statistical_analysis.py
========================
Paired statistical comparison of LOSO cross-validation accuracies for
SpatiotemporalRiemannianMamba vs. BaselineCNN (ds005189, False Memory task).

Run locally:
    pip install numpy scipy
    python statistical_analysis.py
"""

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Input data — 29-fold LOSO accuracies (one value per held-out subject)
# ─────────────────────────────────────────────────────────────────────────────
mamba_per_fold = [
    0.634146, 0.56, 0.4925, 0.5675, 0.57, 0.525, 0.515, 0.4975, 0.515, 0.575,
    0.5275, 0.6025, 0.5075, 0.5725, 0.55, 0.6475, 0.495, 0.685, 0.605, 0.5775,
    0.6375, 0.51, 0.5325, 0.5, 0.5775, 0.5975, 0.5, 0.505, 0.5225,
]

cnn_per_fold = [
    0.521951, 0.54, 0.4525, 0.555, 0.5525, 0.62, 0.54, 0.485, 0.495, 0.59,
    0.545, 0.6225, 0.5, 0.5325, 0.5275, 0.5125, 0.4925, 0.465, 0.5, 0.57,
    0.465, 0.5175, 0.495, 0.4725, 0.5, 0.51, 0.53, 0.505, 0.5075,
]


def run_paired_analysis(model_a_name, model_a_scores, model_b_name, model_b_scores, alpha=0.05):
    """
    Compute descriptive stats, paired t-test, Cohen's d, and the 95% CI
    of the mean difference between two paired sets of fold accuracies.
    """
    a = np.asarray(model_a_scores, dtype=np.float64)
    b = np.asarray(model_b_scores, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"Fold count mismatch: {model_a_name} has {a.shape[0]} folds, "
            f"{model_b_name} has {b.shape[0]} folds. LOSO arrays must be paired 1:1."
        )

    n = a.shape[0]
    diff = a - b

    # Descriptive statistics (sample std, ddof=1)
    mean_a, std_a = np.mean(a), np.std(a, ddof=1)
    mean_b, std_b = np.mean(b), np.std(b, ddof=1)
    mean_diff, std_diff = np.mean(diff), np.std(diff, ddof=1)

    # Paired (related-samples) t-test
    t_stat, p_value = stats.ttest_rel(a, b)

    # Cohen's d for paired samples (standardized mean difference)
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("nan")

    # 95% CI on the mean difference (t-distribution, df = n-1)
    sem_diff = stats.sem(diff)
    ci_low, ci_high = stats.t.interval(1 - alpha, df=n - 1, loc=mean_diff, scale=sem_diff)

    return {
        "n_folds": n,
        "mean_a": mean_a, "std_a": std_a,
        "mean_b": mean_b, "std_b": std_b,
        "mean_diff": mean_diff, "std_diff": std_diff,
        "t_stat": t_stat, "p_value": p_value, "df": n - 1,
        "cohens_d": cohens_d,
        "ci_low": ci_low, "ci_high": ci_high,
        "alpha": alpha,
    }


def print_report(model_a_name, model_b_name, results):
    sep = "=" * 64
    sig_flag = "YES" if results["p_value"] < results["alpha"] else "NO"

    print(f"\n{sep}")
    print("  PAIRED STATISTICAL ANALYSIS — LOSO CROSS-VALIDATION")
    print(sep)
    print(f"  Folds (paired subjects) : {results['n_folds']}")
    print(f"  Significance level (α)  : {results['alpha']}")

    print(f"\n  {'Model':<20}{'Mean Acc.':>12}{'Std Dev.':>12}")
    print(f"  {'-'*44}")
    print(f"  {model_a_name:<20}{results['mean_a']*100:>11.4f}%{results['std_a']*100:>11.4f}%")
    print(f"  {model_b_name:<20}{results['mean_b']*100:>11.4f}%{results['std_b']*100:>11.4f}%")

    print(f"\n  {'Paired difference (A - B)':<32}")
    print(f"  {'-'*44}")
    print(f"  Mean difference          : {results['mean_diff']*100:.4f} pp")
    print(f"  Std of difference        : {results['std_diff']*100:.4f} pp")

    print(f"\n  Paired t-test (scipy.stats.ttest_rel)")
    print(f"  {'-'*44}")
    print(f"  t-statistic              : {results['t_stat']:.6f}")
    print(f"  p-value (two-tailed)     : {results['p_value']:.6f}")
    print(f"  degrees of freedom       : {results['df']}")
    print(f"  Statistically significant: {sig_flag} (p {'<' if sig_flag=='YES' else '>='} {results['alpha']})")

    print(f"\n  Effect Size")
    print(f"  {'-'*44}")
    print(f"  Cohen's d (paired)       : {results['cohens_d']:.4f}")

    print(f"\n  95% Confidence Interval — Mean Difference")
    print(f"  {'-'*44}")
    print(f"  [{results['ci_low']*100:.4f} pp, {results['ci_high']*100:.4f} pp]")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    results = run_paired_analysis(
        "SpatioMamba", mamba_per_fold,
        "BaselineCNN", cnn_per_fold,
    )
    print_report("SpatioMamba", "BaselineCNN", results)