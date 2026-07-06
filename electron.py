#!/usr/bin/env python3
"""Regressione per sciami di elettroni in CRILIN: dalla point cloud delle
celle (7x7x5 cristalli PbF2) a uno scalare per evento, con una rete a
convoluzioni su grafo (EdgeConv / ParticleNet) e loss MSE.

Idea fisica: il punto in cui lo sciame inizia e' scritto nella GEOMETRIA del
deposito - come l'energia si ripartisce tra le celle (profilo longitudinale e
trasverso) - piu' che nella sua scala assoluta. Per questo ogni cella entra
nella rete con posizione, energia digitizzata e frazione di energia
dell'evento; l'energia assoluta resta come contesto perche' la profondita'
dello sciame cresce come ln(E).

Convenzione geometrica: l'asse z e' INVERTITO rispetto al fascio - z grande =
lato fascio (faccia d'ingresso: layer a z=+92.5 mm), lo sciame si sviluppa
verso z decrescenti (layer piu' profondo a z=-95.5 mm). La profondita' e'
quindi (z_fronte - z).

Target (--target): qualunque branch scalare del tree. Il vero primo vertice
longitudinale NON e' negli ntuple attuali (Vertex_z e' il punto di
generazione, costante a 400 mm), quindi la pipeline si valida sulle posizioni
con verita' disponibile - Vertex_x / Vertex_y (spot del fascio, +-2.5 mm),
stesso tipo di task forma->posizione - oppure su PrimaryEnergy. Con i nuovi
ntuple bastera' --target FirstVertex_z.

Ambiente (lxplus, GPU):
  source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh

Esempi:
  python electron.py --file uniform.root --train --epochs 60
  python electron.py --file uniform.root --predict-file fixed99GeV.root \
                     --output prediction_99gev.root
"""

import argparse

import awkward as ak
import numpy as np
import tensorflow as tf
import uproot
from tensorflow.keras import layers

# geometria della matrice CRILIN
N_CELLS = 7 * 7 * 5
TREE_NAME = "events"

# digitizzazione e selezione (come nel training del paper)
CELL_CUT_GEV = 0.05    # soglia per cella dopo lo smearing, taglia le MIP
EVENT_CUT_GEV = 0.5    # energia minima totale per tenere l'evento

OBSERVER_BRANCHES = ["EventID", "PrimaryEnergy", "VD_energy"]
HIT_BRANCHES = ["Hit_x", "Hit_y", "Hit_z", "Hit_E"]


def cli():
    p = argparse.ArgumentParser(
        description="Regressione ParticleNet per sciami di elettroni in CRILIN"
    )
    p.add_argument("--file", required=True,
                   help="ROOT file di training (e di predizione, se --predict-file manca)")
    p.add_argument("--predict-file", default=None,
                   help="ROOT file su cui predire (default: --file)")
    p.add_argument("--target", default="Vertex_x",
                   help="branch scalare da regredire: Vertex_x/Vertex_y "
                        "(posizioni con verita' disponibile), PrimaryEnergy, "
                        "o FirstVertex_z quando esistera'")
    p.add_argument("--target-scale", type=float, default=None,
                   help="fattore applicato al target (default: 1e-3 per le "
                        "branch *Energy*, MeV->GeV; 1 altrimenti, posizioni in mm)")
    p.add_argument("--pe", type=float, default=0.2,
                   help="fotoelettroni per MeV dello smearing Poissoniano "
                        "(resa Cherenkov misurata su questi ntuple: ~24.4)")
    p.add_argument("--train", action="store_true",
                   help="allena la rete (altrimenti carica i pesi)")
    p.add_argument("--weights", default="particlenet_weights.weights.h5",
                   help="file dei pesi da scrivere (train) o leggere (predict)")
    p.add_argument("--output", default="prediction.root",
                   help="ROOT file di output con le predizioni")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-events", type=int, default=None,
                   help="legge al massimo questi eventi (per test rapidi)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# dati: lettura, digitizzazione, costruzione della point cloud
# ---------------------------------------------------------------------------

def _to_matrix(jagged):
    """Vettore jagged per-hit -> matrice (eventi, N_CELLS) float32, zero-padded."""
    filled = ak.fill_none(ak.pad_none(jagged, N_CELLS, clip=True), 0.0)
    return ak.to_numpy(filled).astype(np.float32)


def read_events(path, target, max_events=None):
    """Legge dal tree solo le branch necessarie."""
    wanted = list(dict.fromkeys(OBSERVER_BRANCHES + HIT_BRANCHES + [target]))
    tree = uproot.open(f"{path}:{TREE_NAME}")
    return tree.arrays(wanted, entry_stop=max_events)


def digitize(energy_mev, pe_per_mev):
    """Fotostatistica: conta i fotoelettroni con una Poissoniana e riconverte
    in GeV, poi azzera le celle sotto soglia."""
    photons = np.random.poisson(pe_per_mev * energy_mev)
    energy_gev = photons.astype(np.float32) / pe_per_mev / 1e3
    energy_gev[energy_gev < CELL_CUT_GEV] = 0.0
    return energy_gev


def build_point_cloud(arrays, args):
    """Costruisce (X, y, osservatori, selezione).

    X ha una riga per cella con 5 feature: x, y, z [mm], energia digitizzata
    [GeV] e frazione di energia dell'evento. Le celle sotto soglia sono
    azzerate del tutto (la rete le riconosce come padding).
    """
    pos = [_to_matrix(arrays[b]) for b in ("Hit_x", "Hit_y", "Hit_z")]
    energy = digitize(_to_matrix(arrays["Hit_E"]), args.pe)

    dead = energy == 0.0
    for coord in pos:
        coord[dead] = 0.0

    total = energy.sum(axis=1)
    keep = total > EVENT_CUT_GEV

    share = energy / np.maximum(total, 1e-9)[:, None]  # forma dello sciame
    cloud = np.stack(pos + [energy, share], axis=-1)   # (eventi, N_CELLS, 5)

    y = (ak.to_numpy(arrays[args.target]) * args.target_scale).astype(np.float32)

    observers = {b: ak.to_numpy(arrays[b]).astype(np.float64) for b in OBSERVER_BRANCHES}
    observers["PrimaryEnergy"] *= 1e-3   # GeV
    observers["VD_energy"] *= 1e-3       # GeV
    observers[args.target] = ak.to_numpy(arrays[args.target]) * args.target_scale

    return cloud[keep], y[keep], observers, keep


# ---------------------------------------------------------------------------
# rete: EdgeConv -> pooling globale -> testa densa (struttura ParticleNet)
# ---------------------------------------------------------------------------

class EdgeConvolution(layers.Layer):
    """Blocco EdgeConv: per ogni cella prende i k vicini piu' prossimi nello
    spazio delle feature, applica una MLP condivisa alle coppie
    (centro, vicino - centro) e aggrega con il massimo sui vicini.

    Le MLP puntuali sono Dense sull'ultimo asse del tensore (B, N, k, 2F):
    equivalenti a convoluzioni 1x1 ma applicate direttamente alle feature di
    ogni lato del grafo.
    """

    def __init__(self, neighbours, width, **kwargs):
        super().__init__(**kwargs)
        self.k = neighbours
        self.width = width
        self.mlp_a = layers.Dense(width, activation="relu")
        self.norm_a = layers.BatchNormalization()
        self.mlp_b = layers.Dense(width, activation="relu")
        self.norm_b = layers.BatchNormalization()

    def call(self, feats):
        # una cella e' reale se ha almeno una feature non nulla
        valid = tf.cast(tf.reduce_any(feats != 0.0, axis=-1), feats.dtype)  # (B, N)

        # distanze quadre per differenza diretta (broadcasting sulle coppie)
        delta = feats[:, :, None, :] - feats[:, None, :, :]
        dist2 = tf.reduce_sum(delta * delta, axis=-1)                       # (B, N, N)

        # le coppie con una cella di padding vengono spinte a distanza enorme
        both_real = valid[:, :, None] * valid[:, None, :]
        dist2 += (1.0 - both_real) * 1e6

        # indici dei k vicini (il piu' vicino e' la cella stessa)
        knn = tf.math.top_k(-dist2, k=self.k).indices                       # (B, N, k)
        neigh = tf.gather(feats, knn, batch_dims=1)                         # (B, N, k, F)
        neigh *= valid[:, :, None, None]  # centri di padding -> vicini nulli

        centre = tf.repeat(feats[:, :, None, :], self.k, axis=2)
        edges = tf.concat([centre, neigh - centre], axis=-1)                # (B, N, k, 2F)

        h = self.norm_a(self.mlp_a(edges))
        h = self.norm_b(self.mlp_b(h))
        return tf.reduce_max(h, axis=2)                                     # (B, N, width)


def make_particlenet(n_cells, n_features):
    """Tre blocchi EdgeConv (k=50; 32, 32, 64 canali), media globale sulle
    celle, testa densa 128 -> 64 -> 1 con BatchNorm e Dropout 0.3."""
    cloud = layers.Input(shape=(n_cells, n_features))

    h = EdgeConvolution(50, 32)(cloud)
    h = EdgeConvolution(50, 32)(h)
    h = EdgeConvolution(50, 64)(h)

    h = layers.GlobalAveragePooling1D()(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.BatchNormalization()(h)
    guess = layers.Dense(1)(h)

    return tf.keras.Model(cloud, guess)


# ---------------------------------------------------------------------------
# training e predizione
# ---------------------------------------------------------------------------

def fit(model, cloud, y, args):
    print(f"training su {cloud.shape[0]} eventi, target={args.target}")
    model.summary()
    best = tf.keras.callbacks.ModelCheckpoint(
        args.weights, monitor="val_loss",
        save_best_only=True, save_weights_only=True, verbose=1,
    )
    model.fit(cloud, y,
              batch_size=args.batch_size, epochs=args.epochs,
              validation_split=0.2, shuffle=True, callbacks=[best])
    model.load_weights(args.weights)  # riparte dall'epoca migliore


def predict_and_save(model, args):
    source = args.predict_file or args.file
    arrays = read_events(source, args.target, args.max_events)
    cloud, y, observers, keep = build_point_cloud(arrays, args)
    print(f"predizione su {cloud.shape[0]} eventi da {source}")

    guess = model.predict(cloud, batch_size=args.batch_size).flatten()

    spread = np.std((guess - y) / y) if np.all(y != 0) else float("nan")
    q16, q50, q84 = np.percentile(guess - y, [16, 50, 84])
    print(f"std((pred-vero)/vero) = {spread:.4g}")
    print(f"bias (mediana) = {q50:.4g}   sigma_eff(16-84) = {0.5 * (q84 - q16):.4g}")

    # predizione riallineata a TUTTI gli eventi del file (0 = evento scartato)
    aligned = np.zeros(len(keep))
    aligned[keep] = guess

    record = dict(observers)
    record[f"pred_{args.target}"] = aligned
    record["selected"] = keep.astype(np.int32)

    with uproot.recreate(args.output) as out:
        out[TREE_NAME] = record
    print(f"scritto {args.output} (branch pred_{args.target})")


def main():
    tf.config.set_visible_devices(tf.config.list_physical_devices("GPU"), "GPU")
    args = cli()
    if args.target_scale is None:
        args.target_scale = 1e-3 if "Energy" in args.target else 1.0

    model = make_particlenet(N_CELLS, len(HIT_BRANCHES) + 1)
    # jit_compile=False: l'autotuner XLA/cuDNN fallisce sulle convoluzioni
    # fuse di EdgeConv con TF 2.21 + cuDNN 9.3 sulla T4 di lxplus
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse",
                  metrics=["mae"], jit_compile=False)

    if args.train:
        arrays = read_events(args.file, args.target, args.max_events)
        cloud, y, _, _ = build_point_cloud(arrays, args)
        fit(model, cloud, y, args)
    else:
        print(f"carico i pesi da {args.weights}")
        model.load_weights(args.weights)

    predict_and_save(model, args)


if __name__ == "__main__":
    main()
