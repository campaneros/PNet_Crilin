#!/usr/bin/env python3
"""
Calcola center/scale per hit_x, hit_y, hit_z, log_nph e stampa le righe
gia' pronte da incollare nella sezione `pf_features` del YAML weaver.

weaver applica  (x - subtract_by) * multiply_by, quindi:
  subtract_by = media
  multiply_by = 1 / std

Uso:
  python compute_norm.py uniform_2.5mmsquared_30_100GeV_4mmAldesign_cat.root
"""
import sys

import awkward as ak
import numpy as np
import uproot


def main(path, tree="events", nmax_events=50000):
    br = ["Hit_x", "Hit_y", "Hit_z", "Hit_NCherenkov"]
    arr = uproot.open(f"{path}:{tree}").arrays(
        br, entry_stop=nmax_events, library="ak")

    sel = arr["Hit_NCherenkov"] > 0
    x = ak.flatten(arr["Hit_x"][sel]).to_numpy()
    y = ak.flatten(arr["Hit_y"][sel]).to_numpy()
    z = ak.flatten(arr["Hit_z"][sel]).to_numpy()
    lnph = np.log1p(ak.flatten(arr["Hit_NCherenkov"][sel]).to_numpy())

    def line(name, v):
        m, s = float(v.mean()), float(v.std())
        s = s if s > 0 else 1.0
        return f"      - [{name}, {m:.4g}, {1.0/s:.4g}]"

    print("# incolla in inputs.pf_features.vars:")
    print(line("hit_x", x))
    print(line("hit_y", y))
    print(line("hit_z", z))
    print(line("log_nph", lnph))
    print()
    print(f"# (statistiche su {len(x)} hit di {nmax_events} eventi)")
    print(f"# z: mean={z.mean():.2f} std={z.std():.2f} "
          f"min={z.min():.2f} max={z.max():.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: python compute_norm.py file.root")
    main(sys.argv[1])
