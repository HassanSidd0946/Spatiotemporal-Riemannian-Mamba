"""
step2_0_euclidean_alignment.py

Condition 3 component: Euclidean Alignment (EA) for cross-subject EEG covariance
alignment, per He & Wu (2019) "Transfer Learning for Brain-Computer Interfaces:
A Euclidean Space Data Alignment Approach."

Design constraints satisfied:
  - Fully vectorized (no per-trial Python loops for the matrix multiply)
  - Numerically stable inverse square root via eigendecomposition + epsilon reg
  - Strictly per-subject statistics -> zero leakage across subjects
  - Works whether you align in raw signal-space (X) or covariance-space (C)
"""

import numpy as np


class EuclideanAlignment:
    """
    Computes the subject-level reference covariance R_bar_s and its inverse
    square root R_bar_s^{-1/2}, then aligns that subject's trials.

    IMPORTANT (leakage semantics):
      This is an *unsupervised, per-subject* operator. It does not use labels,
      and it does not use any other subject's trials. That means it is safe
      to fit-and-apply independently on:
        - each of the 28 training subjects (using only their own trials), and
        - the 1 held-out test subject (using only ITS OWN trials, unlabeled)
      This mirrors how EA is used in real BCI transfer-learning deployments:
      a new subject's own unlabeled calibration/session data is used to
      compute R_bar for that subject alone. No information crosses the
      train/test subject boundary.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.R_bar_: np.ndarray | None = None
        self.R_inv_sqrt_: np.ndarray | None = None

    @staticmethod
    def _reference_covariance(X: np.ndarray) -> np.ndarray:
        """
        X : (M, C, T)  trials for a SINGLE subject
        returns R_bar : (C, C) = mean over trials of X_i X_i^T / (T - 1)
        """
        M, C, T = X.shape
        # Vectorized batched covariance: (M, C, T) @ (M, T, C) -> (M, C, C)
        cov = np.einsum('mct,mdt->mcd', X, X) / (T - 1)
        R_bar = cov.mean(axis=0)  # (C, C)
        return R_bar

    def _inv_sqrt(self, R_bar: np.ndarray) -> np.ndarray:
        """
        Numerically stable R_bar^{-1/2} via eigendecomposition.
        R_bar is symmetric PSD -> use eigh (not eig/svd) for stability + speed.
        """
        C = R_bar.shape[0]
        R_reg = R_bar + self.eps * np.eye(C, dtype=R_bar.dtype)
        eigvals, eigvecs = np.linalg.eigh(R_reg)   # ascending order
        eigvals = np.clip(eigvals, a_min=self.eps, a_max=None)  # guard tiny/neg
        inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
        R_inv_sqrt = (eigvecs * inv_sqrt_vals) @ eigvecs.T
        # Symmetrize to kill float round-off asymmetry
        R_inv_sqrt = 0.5 * (R_inv_sqrt + R_inv_sqrt.T)
        return R_inv_sqrt

    def fit(self, X_subject: np.ndarray) -> "EuclideanAlignment":
        """
        X_subject : (M, C, T) — ALL trials belonging to ONE subject only.
        """
        self.R_bar_ = self._reference_covariance(X_subject)
        self.R_inv_sqrt_ = self._inv_sqrt(self.R_bar_)
        return self

    def transform(self, X_subject: np.ndarray) -> np.ndarray:
        """
        Applies X_tilde_i = R_bar^{-1/2} @ X_i to every trial of the subject
        this operator was fit on. Fully vectorized via einsum.

        X_subject : (M, C, T)
        returns   : (M, C, T) aligned trials
        """
        if self.R_inv_sqrt_ is None:
            raise RuntimeError("Call fit() before transform().")
        return np.einsum('dc,mct->mdt', self.R_inv_sqrt_, X_subject)

    def fit_transform(self, X_subject: np.ndarray) -> np.ndarray:
        self.fit(X_subject)
        return self.transform(X_subject)

    def transform_covariance(self, C_subject: np.ndarray) -> np.ndarray:
        """
        Optional: align already-computed covariance matrices instead of raw
        signals:  C_tilde_i = R_bar^{-1/2} @ C_i @ R_bar^{-1/2}
        C_subject : (M, C, C)
        """
        if self.R_inv_sqrt_ is None:
            raise RuntimeError("Call fit() before transform_covariance().")
        R = self.R_inv_sqrt_
        return np.einsum('dc,mce,ef->mdf', R, C_subject, R)


def align_dataset_per_subject(X: np.ndarray, subjects: np.ndarray, eps: float = 1e-6):
    """
    Convenience driver for the FULL dataset (all 29 subjects), applying EA
    independently to each subject's own trial pool. This is what you call
    once, up front, on the entire X_np/subjects_np arrays -- it is NOT where
    leakage would be introduced, because each subject's R_bar only ever sees
    that subject's own trials, train or test.

    X        : (N, C, T)  full dataset, all subjects concatenated
    subjects : (N,)       subject id per trial
    returns  : X_aligned  (N, C, T), same ordering as input
    """
    X_aligned = np.empty_like(X, dtype=np.float32)
    for sub in np.unique(subjects):
        mask = subjects == sub
        ea = EuclideanAlignment(eps=eps)
        X_aligned[mask] = ea.fit_transform(X[mask]).astype(np.float32)
    return X_aligned


# =============================================================================
# Self-test — runs only when this file is executed directly
#   (`python step2_0_euclidean_alignment.py`), same pattern as
#   step2_1_covariance_tangentspace.py
# =============================================================================
if __name__ == "__main__":

    print(f"\n{'='*60}")
    print("  Euclidean Alignment — Self-Test")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    N_SUBJECTS = 3
    N_TRIALS_PER_SUB = 40
    N_CHANNELS = 62
    N_TIMES = 251

    # Mock multi-subject dataset: each subject gets its own random SPD
    # "distortion" matrix to mimic inter-subject non-stationarity.
    X_list, subj_list = [], []
    for sub_id in range(N_SUBJECTS):
        A = rng.standard_normal((N_CHANNELS, N_CHANNELS))
        distortion = A @ A.T + 2 * np.eye(N_CHANNELS)
        raw_trials = rng.standard_normal((N_TRIALS_PER_SUB, N_CHANNELS, N_TIMES))
        X_sub = np.einsum('cd,mdt->mct', distortion, raw_trials)
        X_list.append(X_sub)
        subj_list.append(np.full(N_TRIALS_PER_SUB, sub_id))

    X_mock = np.concatenate(X_list, axis=0).astype(np.float32)
    subjects_mock = np.concatenate(subj_list, axis=0)

    print(f"\n  X_mock shape    : {X_mock.shape}  [n_trials, n_channels, n_times]")
    print(f"  Subjects        : {sorted(np.unique(subjects_mock).tolist())}")

    print(f"\n{'='*60}")
    print("  Applying Euclidean Alignment (per subject)")
    print(f"{'='*60}")

    X_aligned = align_dataset_per_subject(X_mock, subjects_mock, eps=1e-6)
    print(f"  X_aligned shape : {X_aligned.shape}")

    print(f"\n{'='*60}")
    print("  Verification — mean covariance per subject should ≈ Identity")
    print(f"{'='*60}\n")
    print(f"  {'Subject':<10} {'max|R_mean - I|':<20} {'Pass?'}")
    print(f"  {'-'*45}")

    all_pass = True
    for sub_id in sorted(np.unique(subjects_mock)):
        mask = subjects_mock == sub_id
        X_sub_aligned = X_aligned[mask]
        T = X_sub_aligned.shape[-1]
        cov_aligned = np.einsum('mct,mdt->mcd', X_sub_aligned, X_sub_aligned) / (T - 1)
        R_mean = cov_aligned.mean(axis=0)
        err = float(np.max(np.abs(R_mean - np.eye(N_CHANNELS))))
        ok = err < 1e-3
        all_pass &= ok
        print(f"  {sub_id:<10} {err:<20.6e} {'✓' if ok else '✗'}")

    assert all_pass, "One or more subjects failed to whiten to Identity!"

    print(f"\n{'='*60}")
    print("  ✓ Self-test passed — EA correctly whitens each subject's")
    print("    mean spatial covariance to the Identity matrix.")
    print(f"{'='*60}\n")