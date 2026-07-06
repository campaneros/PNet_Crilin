#!/usr/bin/env python3
"""
Suddivide un ROOT grande in N file piu' piccoli: il data loader di
weaver lavora meglio con molti file (li distribuisce fra i worker e
mescola gli eventi leggendone un chunk per volta).

Esempio (train/val da spettro uniforme + test da 99 GeV):
  python split_files.py uniform_..._cat.root  -o files/train --nfiles 40
  python split_files.py fixed99GeV_....root    -o files/test  --nfiles 10

Poi in weaver:
  --data-train 'files/train/*.root'
  --data-test  'files/test/*.root'
(weaver tiene automaticamente una frazione di train come validation
con --train-val-split, default 0.8).
"""
import argparse
import os

import uproot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--tree", default="events")
    ap.add_argument("--nfiles", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    f = uproot.open(f"{args.input}:{args.tree}")
    n = f.num_entries
    branches = f.keys()
    per = (n + args.nfiles - 1) // args.nfiles
    base = os.path.splitext(os.path.basename(args.input))[0]
    print(f"{n} eventi -> {args.nfiles} file da ~{per} eventi")

    for i in range(args.nfiles):
        lo, hi = i * per, min((i + 1) * per, n)
        if lo >= hi:
            break
        data = f.arrays(branches, entry_start=lo, entry_stop=hi,
                        library="ak")
        out = os.path.join(args.outdir, f"{base}_part{i:03d}.root")
        with uproot.recreate(out) as fo:
            fo[args.tree] = data
        print(f"  {out}: eventi {lo}-{hi}")


if __name__ == "__main__":
    main()
