#!/usr/bin/env python3
"""Stima del punto di interazione longitudinale dello sciame SENZA verita' MC,
usando solo il deposito misurato: le frazioni di energia per layer pesate
lungo z.

Due stimatori per evento:

1. **baricentro longitudinale calibrato**
   <z> = sum(E_l * z_l) / sum(E_l) sui 5 layer. La profondita' media di uno
   sciame EM cresce come ln(E), quindi si fitta <z> vs ln(E_misurata) sul
   campione e si prende il residuo Delta<z> = <z> - atteso(E) come
   spostamento evento-per-evento dell'inizio dello sciame.
   Delta<z> > 0: deposito spostato verso il fascio -> interazione piu'
   precoce (piu' vicina alla faccia d'ingresso).

2. **estrapolazione del profilo Longo-Sestili**
   le 5 frazioni longitudinali f_l vengono fittate con il profilo
   dE/dt ~ (b(t-t0))^(a-1) exp(-b(t-t0)), con t la profondita' in X0 dalla
   faccia d'ingresso: t0 e' direttamente il punto di inizio dello sciame
   (integrale del profilo sui limiti fisici di ogni cristallo).

Convenzione: asse z invertito (z grande = lato fascio). Faccia d'ingresso a
z = +112.5 mm, profondita' t = (112.5 - z)/X0 con X0(PbF2) = 9.37 mm.

Esempi:
  python shower_start.py --file fixed99GeV.root --max-events 20000
  python shower_start.py --file uniform.root --outdir plots_start_uniform
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.optimize import curve_fit
from scipy.special import gammainc

from electron import EVENT_CUT_GEV, TREE_NAME, _to_matrix, digitize

X0_MM = 9.37                 # lunghezza di radiazione del PbF2
CRYSTAL_HALF_MM = 20.0       # cristalli lunghi 4 cm
LAYER_Z = np.array([92.5, 45.5, -1.5, -48.5, -95.5])   # centri, fronte -> fondo
FRONT_Z = LAYER_Z[0] + CRYSTAL_HALF_MM                 # faccia d'ingresso, mm

# limiti dei cristalli in profondita' [X0] (i gap di Al non campionano)
T_IN = (FRONT_Z - (LAYER_Z + CRYSTAL_HALF_MM)) / X0_MM
T_OUT = (FRONT_Z - (LAYER_Z - CRYSTAL_HALF_MM)) / X0_MM


def cli():
    p = argparse.ArgumentParser(
        description="Stimatori del punto di inizio sciame dalle frazioni di energia lungo z"
    )
    p.add_argument("--file", required=True, help="ROOT file di input")
    p.add_argument("--pe", type=float, default=0.2,
                   help="fotoelettroni per MeV dello smearing Poissoniano")
    p.add_argument("--b", type=float, default=0.5,
                   help="parametro b (fisso) del profilo Longo-Sestili [1/X0]")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--skip-profile-fit", action="store_true",
                   help="salta il fit Longo-Sestili evento per evento (lento)")
    p.add_argument("--output", default="shower_start.root")
    p.add_argument("--outdir", default="plots_start")
    return p.parse_args()


def layer_energies(path, pe, max_events):
    """Energia digitizzata per layer, (eventi, 5) GeV, piu' gli osservatori."""
    tree = uproot.open(f"{path}:{TREE_NAME}")
    arrays = tree.arrays(["EventID", "PrimaryEnergy", "Hit_z", "Hit_E"],
                         entry_stop=max_events)
    z = _to_matrix(arrays["Hit_z"])
    E = digitize(_to_matrix(arrays["Hit_E"]), pe)

    per_layer = np.stack(
        [np.sum(E * np.isclose(z, zl), axis=1) for zl in LAYER_Z], axis=1
    )
    obs = {
        "EventID": np.asarray(arrays["EventID"], dtype=np.float64),
        "PrimaryEnergy": np.asarray(arrays["PrimaryEnergy"]) * 1e-3,  # GeV
    }
    return per_layer, obs


def barycenter_shift(E_layers):
    """Stimatore 1: <z> e il suo residuo rispetto all'andamento in ln(E)."""
    E_sum = E_layers.sum(axis=1)
    zbar = E_layers @ LAYER_Z / E_sum
    ln_e = np.log(E_sum)

    if ln_e.std() > 0.01:                       # spettro largo: togli il trend
        coeff = np.polyfit(ln_e, zbar, 1)
    else:                                       # energia fissa: solo la media
        coeff = np.array([0.0, zbar.mean()])
    shift = zbar - np.polyval(coeff, ln_e)
    return zbar, shift, coeff, E_sum


def longo_sestili_t0(fractions, energy_gev, b):
    """Stimatore 2: fit del profilo per un singolo evento -> (t0, a) [X0]."""

    def model(_, t0, a):
        upper = gammainc(a, np.clip(b * (T_OUT - t0), 0.0, None))
        lower = gammainc(a, np.clip(b * (T_IN - t0), 0.0, None))
        frac = upper - lower
        total = frac.sum()
        return frac / total if total > 0 else frac

    # inizializzazione dalla shower theory: a = 1 + b*(ln y + C_e), C_e = -0.5
    y = energy_gev * 1e3 / 9.6                  # E_c(PbF2) = 9.6 MeV
    a_start = max(1.2, 1.0 + b * (np.log(y) - 0.5))
    try:
        par, _ = curve_fit(model, np.arange(5), fractions,
                           p0=[0.3, a_start],
                           bounds=([-3.0, 1.05], [15.0, 40.0]), maxfev=200)
        return par[0], par[1]
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def sigma_eff(x):
    q16, q84 = np.percentile(x, [16, 84])
    return 0.5 * (q84 - q16)


def main():
    import os
    args = cli()

    E_layers, obs = layer_energies(args.file, args.pe, args.max_events)
    keep = E_layers.sum(axis=1) > EVENT_CUT_GEV
    E_layers = E_layers[keep]
    obs = {k: v[keep] for k, v in obs.items()}
    print(f"{len(E_layers)} eventi selezionati da {args.file}")

    zbar, shift, coeff, E_sum = barycenter_shift(E_layers)
    fractions = E_layers / E_sum[:, None]

    print(f"calibrazione <z> = {coeff[1]:.2f} + ({coeff[0]:.2f})*ln(E) [mm]")
    print(f"<z>: media = {zbar.mean():.2f} mm   sigma_eff = {sigma_eff(zbar):.2f} mm")
    print(f"Delta<z> (inizio sciame, relativo): sigma_eff = {sigma_eff(shift):.2f} mm"
          f" = {sigma_eff(shift) / X0_MM:.2f} X0")

    record = {
        "EventID": obs["EventID"],
        "PrimaryEnergy": obs["PrimaryEnergy"],
        "E_sum": E_sum.astype(np.float64),
        "zbar": zbar.astype(np.float64),
        "zbar_shift": shift.astype(np.float64),
    }
    for l in range(5):
        record[f"frac_layer{l}"] = fractions[:, l].astype(np.float64)

    if not args.skip_profile_fit:
        fitted = np.array([longo_sestili_t0(f, e, args.b)
                           for f, e in zip(fractions, E_sum)])
        t0, a_par = fitted[:, 0], fitted[:, 1]
        good = np.isfinite(t0)
        z0 = FRONT_Z - t0 * X0_MM               # coordinata z di inizio sciame
        print(f"fit Longo-Sestili riuscito su {good.mean():.1%} degli eventi")
        print(f"t0: mediana = {np.nanmedian(t0):.2f} X0   "
              f"sigma_eff = {sigma_eff(t0[good]):.2f} X0 "
              f"({sigma_eff(t0[good]) * X0_MM:.1f} mm)")
        record["ls_t0"] = t0.astype(np.float64)
        record["ls_a"] = a_par.astype(np.float64)
        record["ls_z0"] = z0.astype(np.float64)

    with uproot.recreate(args.output) as out:
        out[TREE_NAME] = record
    print(f"scritto {args.output}")

    # ------------------------------------------------------------------ plot
    os.makedirs(args.outdir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(zbar, bins=100, histtype="stepfilled", alpha=0.7)
    plt.xlabel("<z> pesato in energia [mm]  (z grande = lato fascio)")
    plt.ylabel("eventi")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/zbar.png", dpi=120)

    plt.figure(figsize=(6, 4))
    lo, hi = np.percentile(shift, [0.5, 99.5])
    plt.hist(shift, bins=100, range=(lo, hi), histtype="stepfilled", alpha=0.7)
    plt.xlabel("Delta<z> rispetto all'atteso(E) [mm]  (>0 = inizio precoce)")
    plt.ylabel("eventi")
    plt.title(f"sigma_eff = {sigma_eff(shift):.2f} mm = {sigma_eff(shift)/X0_MM:.2f} X0")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/zbar_shift.png", dpi=120)

    plt.figure(figsize=(6, 4))
    ln_e = np.log(E_sum)
    plt.hist2d(ln_e, zbar, bins=80, cmin=1)
    xs = np.linspace(ln_e.min(), ln_e.max(), 50)
    plt.plot(xs, np.polyval(coeff, xs), "r--", lw=1.5, label="calibrazione")
    plt.xlabel("ln(E_sum [GeV])")
    plt.ylabel("<z> [mm]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/zbar_vs_lnE.png", dpi=120)

    plt.figure(figsize=(6, 4))
    depth = (FRONT_Z - LAYER_Z) / X0_MM
    plt.errorbar(depth, fractions.mean(axis=0), yerr=fractions.std(axis=0),
                 fmt="o-", capsize=3)
    plt.xlabel("profondita' del layer [X0]")
    plt.ylabel("frazione di energia media")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/profile.png", dpi=120)

    if not args.skip_profile_fit:
        plt.figure(figsize=(6, 4))
        plt.hist(t0[good], bins=100, range=(-3, 8), histtype="stepfilled", alpha=0.7)
        plt.xlabel("t0 Longo-Sestili [X0 dalla faccia d'ingresso]")
        plt.ylabel("eventi")
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/ls_t0.png", dpi=120)

    print(f"plot in {args.outdir}/")


if __name__ == "__main__":
    main()
