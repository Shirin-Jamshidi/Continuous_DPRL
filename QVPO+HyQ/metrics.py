# metrics.py
"""
Shared comparison metrics for ContinuousCartPole experiments.
=============================================================
Drop this file into the same directory as:
  - vanilla_diffusion.py
  - qvpo.py
  - diffusion_ql.py   (QVPO + Hy-Q)

Each of those scripts imports and calls the MetricsTracker the same way,
so results are directly comparable across methods.

Usage (copy-paste into any training script)
-------------------------------------------
    from metrics import MetricsTracker, print_comparison_table

    tracker = MetricsTracker(method_name="QVPO")

    # during training — call after every gradient step:
    tracker.log_step(
        step        = step,
        critic_loss = c_loss,      # pass None if method has no critic
        policy_loss = p_loss,
    )

    # during evaluation — call after every eval rollout:
    tracker.log_eval(
        step    = step,
        returns = [ep1_return, ep2_return, ...],   # list of episode returns
    )

    # at the end — save and optionally print:
    tracker.save("results/qvpo_metrics.npz")
    tracker.print_summary()

    # to compare several saved result files side-by-side:
    print_comparison_table([
        "results/vanilla_diffusion_metrics.npz",
        "results/qvpo_metrics.npz",
        "results/hyq_metrics.npz",
    ])

Metrics recorded
----------------
  Per gradient step  (logged every step)
    step              — gradient step index
    critic_loss       — Bellman MSE  (NaN for VanillaDP which has no critic)
    policy_loss       — BC loss / VLO loss / total policy loss
    wall_time         — seconds since tracker was created

  Per evaluation checkpoint  (logged every eval_interval steps)
    eval_step         — gradient step at eval time
    eval_mean         — mean episodic return over n eval episodes
    eval_std          — std of episodic returns
    eval_min          — worst episode return
    eval_max          — best episode return
    eval_median       — median episodic return
    eval_iqm          — interquartile mean (robust central tendency)
    eval_wall_time    — seconds since tracker creation at eval time

Comparison table columns
------------------------
  Method | Steps | EvalMean±Std | Median | IQM | Best | Worst
"""

import time
import os
from typing import Optional, List

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Lightweight, dependency-free metrics logger.
    All data stored as plain Python lists → saved as .npz.
    """

    def __init__(self, method_name: str):
        self.method_name = method_name
        self._t0         = time.time()

        # step-level logs
        self.steps        : List[int]   = []
        self.critic_losses: List[float] = []   # NaN when no critic
        self.policy_losses: List[float] = []
        self.wall_times   : List[float] = []

        # eval-level logs
        self.eval_steps     : List[int]   = []
        self.eval_means     : List[float] = []
        self.eval_stds      : List[float] = []
        self.eval_mins      : List[float] = []
        self.eval_maxs      : List[float] = []
        self.eval_medians   : List[float] = []
        self.eval_iqms      : List[float] = []
        self.eval_wall_times: List[float] = []

    # ------------------------------------------------------------------
    def log_step(
        self,
        step:         int,
        policy_loss:  float,
        critic_loss:  Optional[float] = None,   # None → logged as NaN
    ):
        self.steps.append(step)
        self.policy_losses.append(float(policy_loss))
        self.critic_losses.append(float(critic_loss) if critic_loss is not None
                                  else float("nan"))
        self.wall_times.append(time.time() - self._t0)

    # ------------------------------------------------------------------
    def log_eval(self, step: int, returns: List[float]):
        arr = np.array(returns, dtype=np.float64)
        self.eval_steps.append(step)
        self.eval_means.append(float(arr.mean()))
        self.eval_stds.append(float(arr.std()))
        self.eval_mins.append(float(arr.min()))
        self.eval_maxs.append(float(arr.max()))
        self.eval_medians.append(float(np.median(arr)))
        self.eval_iqms.append(float(_iqm(arr)))
        self.eval_wall_times.append(time.time() - self._t0)

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            method_name   = np.array([self.method_name]),
            steps         = np.array(self.steps,         dtype=np.int64),
            critic_losses = np.array(self.critic_losses, dtype=np.float64),
            policy_losses = np.array(self.policy_losses, dtype=np.float64),
            wall_times    = np.array(self.wall_times,    dtype=np.float64),
            eval_steps      = np.array(self.eval_steps,       dtype=np.int64),
            eval_means      = np.array(self.eval_means,       dtype=np.float64),
            eval_stds       = np.array(self.eval_stds,        dtype=np.float64),
            eval_mins       = np.array(self.eval_mins,        dtype=np.float64),
            eval_maxs       = np.array(self.eval_maxs,        dtype=np.float64),
            eval_medians    = np.array(self.eval_medians,     dtype=np.float64),
            eval_iqms       = np.array(self.eval_iqms,        dtype=np.float64),
            eval_wall_times = np.array(self.eval_wall_times,  dtype=np.float64),
        )
        # print(f"  [MetricsTracker] Saved → {path}")

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print a one-block summary for this method."""
        n_steps = len(self.steps)
        if n_steps == 0:
            print(f"  [{self.method_name}] No steps logged yet.")
            return

        # Training losses (last 10 % of steps)
        tail = max(1, n_steps // 10)
        avg_p = float(np.nanmean(self.policy_losses[-tail:]))
        avg_c = float(np.nanmean(self.critic_losses[-tail:]))
        wall  = self.wall_times[-1] if self.wall_times else 0.0

        print(f"\n  ┌─ {self.method_name} ─ Summary ─────────────────────────────")
        print(f"  │  Total steps        : {n_steps:,}")
        print(f"  │  Wall time          : {wall/60:.1f} min")
        print(f"  │  Avg policy loss    : {avg_p:.6f}  (last {tail} steps)")
        if not np.isnan(avg_c):
            print(f"  │  Avg critic loss    : {avg_c:.6f}  (last {tail} steps)")

        if self.eval_means:
            best_mean = max(self.eval_means)
            last_mean = self.eval_means[-1]
            last_std  = self.eval_stds[-1]
            last_iqm  = self.eval_iqms[-1]
            print(f"  │  Evals recorded     : {len(self.eval_means)}")
            print(f"  │  Final eval mean    : {last_mean:.1f} ± {last_std:.1f}")
            print(f"  │  Final eval IQM     : {last_iqm:.1f}")
            print(f"  │  Best eval mean     : {best_mean:.1f}")
        print(f"  └────────────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison table (loads multiple .npz files)
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(npz_paths: List[str]):
    """
    Load several saved MetricsTracker .npz files and print a side-by-side
    comparison table using the FINAL evaluation checkpoint of each method.

    Args
        npz_paths : list of paths to .npz files saved by MetricsTracker.save()

    Example
    -------
        print_comparison_table([
            "results/vanilla_diffusion_metrics.npz",
            "results/qvpo_metrics.npz",
            "results/hyq_metrics.npz",
        ])
    """
    rows = []
    for path in npz_paths:
        if not os.path.exists(path):
            print(f"  [comparison] File not found, skipping: {path}")
            continue
        d = np.load(path, allow_pickle=True)
        name = str(d["method_name"][0])

        eval_means = d["eval_means"]
        if len(eval_means) == 0:
            print(f"  [comparison] No eval data in {path}, skipping.")
            continue

        # Use the final eval checkpoint
        rows.append({
            "method"   : name,
            "steps"    : int(d["steps"][-1]) if len(d["steps"]) else 0,
            "mean"     : float(eval_means[-1]),
            "std"      : float(d["eval_stds"][-1]),
            "median"   : float(d["eval_medians"][-1]),
            "iqm"      : float(d["eval_iqms"][-1]),
            "best"     : float(d["eval_maxs"][-1]),
            "worst"    : float(d["eval_mins"][-1]),
            "wall_min" : float(d["wall_times"][-1]) / 60.0
                         if len(d["wall_times"]) else 0.0,
            "best_ever": float(np.max(eval_means)),
        })

    if not rows:
        print("  [comparison] Nothing to compare.")
        return

    # ── Table header ──────────────────────────────────────────────────────────
    col_w = {
        "method" : max(len(r["method"]) for r in rows) + 2,
        "steps"  : 9,
        "mean"   : 14,
        "median" : 8,
        "iqm"    : 8,
        "best"   : 7,
        "worst"  : 7,
        "peak"   : 8,
        "time"   : 9,
    }

    def _c(s, w):  return str(s).center(w)
    def _r(s, w):  return str(s).rjust(w)

    sep   = "─" * (sum(col_w.values()) + len(col_w) + 1)
    hdr   = (f"│{_c('Method', col_w['method'])}│"
             f"{_c('Steps', col_w['steps'])}│"
             f"{_c('Mean ± Std', col_w['mean'])}│"
             f"{_c('Median', col_w['median'])}│"
             f"{_c('IQM', col_w['iqm'])}│"
             f"{_c('Best', col_w['best'])}│"
             f"{_c('Worst', col_w['worst'])}│"
             f"{_c('PeakMean', col_w['peak'])}│"
             f"{_c('Time(m)', col_w['time'])}│")

    print(f"\n  ┌{sep}┐")
    print(f"  {hdr}")
    print(f"  ├{sep}┤")

    # Sort by final eval mean descending
    for r in sorted(rows, key=lambda x: x["mean"], reverse=True):
        mean_std = f"{r['mean']:.1f} ± {r['std']:.1f}"
        median = f"{r['median']:.1f}"
        iqm = f"{r['iqm']:.1f}"
        best = f"{r['best']:.0f}"
        worst = f"{r['worst']:.0f}"
        peak = f"{r['best_ever']:.1f}"
        wall_min = f"{r['wall_min']:.1f}"
        line = (f"│{_c(r['method'],  col_w['method'])}│"
                f"{_r(str(r['steps']), col_w['steps'])}│"
                f"{_c(mean_std,       col_w['mean'])}│"
            f"{_r(median,        col_w['median'])}│"
            f"{_r(iqm,           col_w['iqm'])}│"
            f"{_r(best,          col_w['best'])}│"
            f"{_r(worst,         col_w['worst'])}│"
            f"{_r(peak,          col_w['peak'])}│"
            f"{_r(wall_min,      col_w['time'])}│")
        print(f"  {line}")

    print(f"  └{sep}┘")
    print(f"\n  Columns: Mean±Std and Median/IQM are from the FINAL eval checkpoint.")
    print(f"  PeakMean = best eval mean ever recorded during training.")
    print(f"  IQM      = interquartile mean (robust to outlier episodes).\n")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iqm(arr: np.ndarray) -> float:
    """Interquartile mean: mean of values between 25th and 75th percentile."""
    q25, q75 = np.percentile(arr, [25, 75])
    mask = (arr >= q25) & (arr <= q75)
    return float(arr[mask].mean()) if mask.any() else float(arr.mean())



print_comparison_table([
    "vanilla_diffusion_metrics.npz",
    "qvpo+hy-q_metrics.npz",
    "qvpo_metrics.npz",
])
