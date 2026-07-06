#!/usr/bin/env python3
"""Evaluate a prediction.root produced by electron_vertex.py predict.

Computes the residuals pred - truth, their bias (median) and effective
resolution sigma_eff = (q84 - q16)/2, and draws:
  - the residual distribution,
  - pred vs truth,
  - bias and sigma_eff versus PrimaryEnergy (when the sample spans a range).

Usage:
    python evaluate.py prediction.root [--target PrimaryEnergy] [--outdir plots]
"""

import argparse
import os

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigma_eff(x):
    q16, q84 = np.percentile(x, [16, 84])
    return 0.5 * (q84 - q16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="prediction ROOT file")
    ap.add_argument("--tree", default="events")
    ap.add_argument("--target", default=None,
                    help="target branch name (default: inferred from pred_* branch)")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--energy-bins", type=int, default=7,
                    help="number of PrimaryEnergy bins for the binned plots")
    args = ap.parse_args()

    t = uproot.open(f"{args.input}:{args.tree}")
    names = [b for b in t.keys()]
    target = args.target
    if target is None:
        preds = [b for b in names if b.startswith("pred_")]
        if not preds:
            raise SystemExit("no pred_* branch found; pass --target")
        target = preds[0][len("pred_"):]
    pred = np.asarray(t[f"pred_{target}"].array())
    truth = np.asarray(t[target].array())
    keep = np.asarray(t["selected"].array()) > 0 if "selected" in names \
        else np.ones(len(pred), dtype=bool)
    pred, truth = pred[keep], truth[keep]
    res = pred - truth

    bias, sig = np.median(res), sigma_eff(res)
    print(f"target={target}  events={len(res)}")
    print(f"bias (median)   = {bias:.4g}")
    print(f"sigma_eff(16-84)= {sig:.4g}")
    if np.all(truth > 0):
        print(f"relative sigma_eff = {sigma_eff(res / truth):.4%}")

    os.makedirs(args.outdir, exist_ok=True)

    # residual distribution
    plt.figure(figsize=(6, 4))
    lo, hi = np.percentile(res, [0.5, 99.5])
    plt.hist(res, bins=100, range=(lo, hi), histtype="stepfilled", alpha=0.7)
    plt.axvline(bias, c="k", lw=1)
    plt.xlabel(f"pred - true  [{target}]")
    plt.ylabel("events")
    plt.title(f"bias={bias:.3g}   sigma_eff={sig:.3g}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "residuals.png"), dpi=120)

    # pred vs truth
    plt.figure(figsize=(5, 5))
    plt.hist2d(truth, pred, bins=100, cmin=1)
    lim = [min(truth.min(), pred.min()), max(truth.max(), pred.max())]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel(f"true {target}")
    plt.ylabel(f"pred {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pred_vs_truth.png"), dpi=120)

    # binned bias / resolution vs PrimaryEnergy, only for spectra with a spread
    if "PrimaryEnergy" in names:
        E = np.asarray(t["PrimaryEnergy"].array())[keep]
        if E.std() / max(E.mean(), 1e-9) > 0.01:
            edges = np.linspace(E.min(), E.max(), args.energy_bins + 1)
            centers, biases, sigmas = [], [], []
            for lo_e, hi_e in zip(edges[:-1], edges[1:]):
                sel = (E >= lo_e) & (E < hi_e)
                if sel.sum() < 50:
                    continue
                centers.append(0.5 * (lo_e + hi_e))
                biases.append(np.median(res[sel]))
                sigmas.append(sigma_eff(res[sel]))
            fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
            axes[0].plot(centers, biases, "o-")
            axes[0].axhline(0, c="gray", lw=1, ls="--")
            axes[0].set_ylabel("bias (median)")
            axes[1].plot(centers, sigmas, "o-")
            axes[1].set_ylabel("sigma_eff")
            axes[1].set_xlabel("PrimaryEnergy [GeV]")
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "vs_energy.png"), dpi=120)

    print(f"plots written to {args.outdir}/")


if __name__ == "__main__":
    main()
