#!/usr/bin/env python3
"""Electron shower regression in CRILIN with ParticleNet: infer the
interaction point from the 3D geometry of the shower.

Modelled on CrilinSim/pion.py (the training code of arXiv:2606.05111), which
regresses the pion energy from the (x, y, z, E) point cloud. Here the task is
different - reconstructing a *position* from the shower *shape* - so the
inputs are adapted:
  - per-cell features (x, y, z, E, E/E_sum): the energy FRACTION of each cell
    is added because a position is encoded in how the energy is shared among
    the cells (longitudinal/transverse profile), not in its absolute scale;
    the absolute E is kept because the shower depth grows like ln(E), which
    helps extrapolating back to the start of the shower;
  - the target is a scalar branch (--target). The true longitudinal first-
    interaction vertex is NOT in the current ntuples (Vertex_z is the constant
    generation point, 400 mm); until new ntuples exist the pipeline is
    validated on the positions that ARE known: Vertex_x/Vertex_y (beam spot,
    +-2.5 mm) - a genuine shape->position regression - or on PrimaryEnergy.
    With new ntuples: --target FirstVertex_z.
Geometry convention: the z axis is INVERTED with respect to the beam
direction - large z is the beam side (front face, layer at z=+92.5 mm) and
the shower develops towards decreasing z (deepest layer at z=-95.5 mm).
Depth in the calorimeter is therefore (z_front - z).

Other practical differences from pion.py: training and prediction files can
differ (--file / --predict-file), only the needed branches are read, --train
is a proper flag, --max-events/--epochs/--batch-size for quick tests.

Environment (lxplus, GPU):
  source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh

Examples:
  python electron.py --file uniform.root --train --epochs 60
  python electron.py --file uniform.root --predict-file fixed99GeV.root
"""

import numpy as np
import uproot
import awkward as ak
import argparse
import tensorflow as tf
from tensorflow.keras import layers, Model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Electron shower regression (ParticleNet, from pion.py)"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="root file used for training (and for prediction if --predict-file is not given)"
    )

    parser.add_argument(
        "--predict-file",
        default=None,
        help="root file used for prediction (default: --file)"
    )

    parser.add_argument(
        "--pe",
        type=float,
        default=0.2,  # pe/MeV
        help="photo-electrons per MeV for Poissonian smearing (measured Cherenkov yield of these ntuples: ~24.4)"
    )

    parser.add_argument(
        "--target",
        default="Vertex_x",
        help="scalar branch to regress: Vertex_x/Vertex_y (positions with "
             "available truth), PrimaryEnergy, or FirstVertex_z once available"
    )

    parser.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="multiplied to the target branch (default: 1e-3 for *Energy* "
             "branches, MeV -> GeV; 1 otherwise, positions stay in mm)"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="train the network (otherwise load the weights)"
    )

    parser.add_argument(
        "--weights",
        default="particlenet_weights.weights.h5",
        help="weights file to write (train) or read (predict)"
    )

    parser.add_argument(
        "--output",
        default="prediction.root",
        help="output prediction file"
    )

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="read at most this many events (for quick tests)"
    )

    return parser.parse_args()


# --- ParticleNet implementation reused as-is from CrilinSim/pion.py ---------

class EdgeConv(layers.Layer):
    def __init__(self, k, filters):
        super().__init__()
        self.k = k
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 1, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 1, activation='relu')
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        k = self.k

        # Compute mask from zeros
        mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)  # (B, hits)
        mask = tf.cast(mask, x.dtype)

        # Pairwise distance
        xx = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        pairwise_distance = xx - 2 * tf.matmul(x, x, transpose_b=True) + tf.transpose(xx, [0, 2, 1])
        mask_matrix = mask[:, :, None] * mask[:, None, :]
        pairwise_distance += (1.0 - mask_matrix) * 1e6

        # KNN
        idx = tf.math.top_k(-pairwise_distance, k=k).indices
        neighbors = tf.gather(x, idx, batch_dims=1)

        # Broadcast mask properly
        mask_expanded = tf.expand_dims(mask, axis=2)             # (B, hits, 1)
        mask_expanded = tf.tile(mask_expanded, [1, 1, k])        # (B, hits, k)
        mask_expanded = tf.expand_dims(mask_expanded, axis=-1)   # (B, hits, k, 1)
        neighbors *= mask_expanded

        # Central features
        central = tf.expand_dims(x, 2)
        central = tf.tile(central, [1, 1, k, 1])

        # Edge features
        edge = tf.concat([central, neighbors - central], axis=-1)

        # Conv layers
        h = self.conv1(edge)
        h = self.bn1(h)
        h = self.conv2(h)
        h = self.bn2(h)

        # Max over neighbors
        h = tf.reduce_max(h, axis=2)
        return h


def build_particlenet(maxhits, nfeat=5):

    inputs = layers.Input(shape=(maxhits, nfeat))
    x = EdgeConv(k=50, filters=32)(inputs)
    x = EdgeConv(k=50, filters=32)(x)
    x = EdgeConv(k=50, filters=64)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, output)

    return model


# --- data preparation (pion.py logic, packaged so that it can be applied ----
# --- identically to the training and to the prediction file) ----------------

MAX_HITS = 7 * 7 * 5  # nchannels


def load_and_digitize(path, args):
    """Read one file, apply the pion.py digitization (Poisson smearing +
    50 MeV cell threshold) and the E_sum > 0.5 GeV event selection.

    Returns X (N,245,4), y_target (N,), the observer branches and the
    selection mask over the original events.
    """
    tree = uproot.open(f"{path}:events")
    branches = ["EventID", "PrimaryEnergy", "VD_energy",
                "Hit_x", "Hit_y", "Hit_z", "Hit_E"]
    if args.target not in branches:
        branches.append(args.target)
    arrays = tree.arrays(branches, entry_stop=args.max_events)

    # pad events to same length
    x = ak.to_numpy(ak.fill_none(ak.pad_none(arrays["Hit_x"], MAX_HITS, clip=True), 0)).astype(np.float32)
    y = ak.to_numpy(ak.fill_none(ak.pad_none(arrays["Hit_y"], MAX_HITS, clip=True), 0)).astype(np.float32)
    z = ak.to_numpy(ak.fill_none(ak.pad_none(arrays["Hit_z"], MAX_HITS, clip=True), 0)).astype(np.float32)
    E = ak.to_numpy(ak.fill_none(ak.pad_none(arrays["Hit_E"], MAX_HITS, clip=True), 0)).astype(np.float32)

    # photostatistics smearing and MIP threshold, as in pion.py
    E = (np.random.poisson(args.pe * E) / args.pe).astype(np.float32) / 1e3  # GeV
    underthre_mask = E < 0.05  # to cut the MIPs
    E[underthre_mask] = 0
    x[underthre_mask] = 0
    y[underthre_mask] = 0
    z[underthre_mask] = 0

    E_sum = np.sum(E, axis=1)
    sel = E_sum > 0.5

    # energy fraction of each cell: makes the shower SHAPE explicit,
    # independently of the absolute energy scale (what a position lives in)
    frac = (E / np.maximum(E_sum, 1e-9)[:, None]).astype(np.float32)

    X = np.stack([x, y, z, E, frac], axis=-1)[sel, :, :]  # (N,245,5)
    y_target = (ak.to_numpy(arrays[args.target]) * args.target_scale).astype(np.float32)[sel]

    observers = {
        "EventID": ak.to_numpy(arrays["EventID"]),
        "PrimaryEnergy": ak.to_numpy(arrays["PrimaryEnergy"]) * 1e-3,  # GeV
        "VD_energy": ak.to_numpy(arrays["VD_energy"]) / 1e3,           # GeV
        args.target: ak.to_numpy(arrays[args.target]) * args.target_scale,
    }
    return X, y_target, observers, sel


def main():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus, 'GPU')
    args = parse_args()
    if args.target_scale is None:
        args.target_scale = 1e-3 if "Energy" in args.target else 1.0

    model = build_particlenet(maxhits=MAX_HITS)
    # jit_compile=False: the XLA autotuner fails on the fused 1x1 convolutions
    # of EdgeConv with TF 2.21 + cuDNN 9.3 on the lxplus T4
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse',
                  metrics=['mae'], jit_compile=False)

    if args.train:
        X, y_target, _, _ = load_and_digitize(args.file, args)
        print(f"training: {X.shape[0]} events from {args.file}, target={args.target}")
        model.summary()
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            args.weights,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        model.fit(X, y_target, batch_size=args.batch_size, epochs=args.epochs,
                  validation_split=0.2, shuffle=True, callbacks=[checkpoint])
        # reload the best epoch before predicting
        model.load_weights(args.weights)
    else:
        print(f"Loading weights from {args.weights}")
        model.load_weights(args.weights)

    predict_file = args.predict_file if args.predict_file else args.file
    X, y_target, observers, sel = load_and_digitize(predict_file, args)
    print(f"predicting: {X.shape[0]} events from {predict_file}")

    pred = model.predict(X, batch_size=args.batch_size).flatten()

    resolution = np.std((pred - y_target) / y_target)
    print("Resolution std((pred-true)/true):", resolution)
    q16, q50, q84 = np.percentile(pred - y_target, [16, 50, 84])
    print(f"bias (median) = {q50:.4g}   sigma_eff(16-84) = {0.5*(q84-q16):.4g}")

    # flat array aligned with the original events (0 = event failed selection)
    pred_to_write = np.zeros(len(sel))
    pred_to_write[sel] = pred

    out = dict(observers)
    out[f"pred_{args.target}"] = pred_to_write
    out["selected"] = sel.astype(np.int32)

    with uproot.recreate(args.output) as fout:
        fout["events"] = out
    print(f"written {args.output} (branch pred_{args.target})")


if __name__ == "__main__":
    main()
