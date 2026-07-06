# CRILIN vertex regression con weaver-core (ParticleNet ufficiale)

Setup **canonico**: usa il framework `weaver-core` di Huilin Qu e il
`ParticleNet` ufficiale come backbone (lo stesso codice usato in CMS),
adattato a **regressione** del vertice di inizio sciame dagli hit
Cherenkov. Non riscrivo l'architettura: la importo da weaver e cambio
solo la testa finale (uscita lineare) e la loss (MSE).

## File

| file                              | scopo                                        |
|-----------------------------------|----------------------------------------------|
| `data/crilin_vertex.yaml`         | data config: point cloud hit + target Vertex_z |
| `networks/particle_net_regr.py`   | model config: ParticleNet ufficiale + testa regressione + MSE |
| `utils/compute_norm.py`           | calcola center/scale da mettere nello YAML   |
| `utils/split_files.py`            | esplode i ROOT grandi in tanti file          |

## 1. Installazione

```bash
pip install weaver-core            # installa anche il pacchetto `weaver`
# oppure, per avere i sorgenti modificabili:
# git clone https://github.com/hqucms/weaver-core && cd weaver-core && pip install -e .
```

`networks/particle_net_regr.py` fa `from weaver.nn.model.ParticleNet
import ParticleNet, FeatureConv`: funziona con l'installazione pip.

## 2. Preparazione dati

I due file da 1.1 GB sono meglio spezzati in tanti file piccoli (il
data loader di weaver mescola gli eventi leggendo un chunk per file):

```bash
python utils/split_files.py uniform_2.5mmsquared_30_100GeV_4mmAldesign_cat.root \
    -o files/train --nfiles 40
python utils/split_files.py fixed99GeV_2.5mmsquared_30_100GeV_4mmAldesign.root \
    -o files/test  --nfiles 10
```

Poi calcola le costanti di normalizzazione e **incollale** nella
sezione `pf_features` di `data/crilin_vertex.yaml` (i valori attuali
sono placeholder):

```bash
python utils/compute_norm.py files/train/uniform_2.5mmsquared_30_100GeV_4mmAldesign_cat_part000.root
```

## 3. Training

```bash
weaver \
  --data-train 'files/train/*.root' \
  --data-config data/crilin_vertex.yaml \
  --network-config networks/particle_net_regr.py \
  --model-prefix models/crilin_pn/net \
  --num-workers 4 --fetch-step 0.02 \
  --batch-size 256 --start-lr 1e-3 --num-epochs 40 \
  --optimizer ranger \
  --gpus 0 \
  --train-mode regression
```

Note:
- `--train-mode regression` dice a weaver di trattare i target come
  valori continui (niente accuracy/softmax, usa la loss di `get_loss`).
- `--train-val-split 0.8` (default) tiene il 20% del train come
  validation.
- niente GPU? togli `--gpus 0`; sara' lento, riduci `length` a 512 nel
  YAML e usa `--num-epochs 5` per un primo test.

## 4. Predizione / valutazione

```bash
weaver --predict \
  --data-test 'files/test/*.root' \
  --data-config data/crilin_vertex.yaml \
  --network-config networks/particle_net_regr.py \
  --model-prefix models/crilin_pn/net_best_epoch_state.pt \
  --gpus 0 --batch-size 256 \
  --predict-output output/pred_99gev.root
```

L'output ROOT contiene la predizione del modello insieme agli
`observers` (EventID, PrimaryEnergy, Vertex_x/y/z veri): da lì calcoli
bias e risoluzione σ_eff (percentili 16–84) come nel paper, es.

```python
import uproot, numpy as np
t = uproot.open("output/pred_99gev.root:Events").arrays(library="np")
res = t["output"] - t["Vertex_z"]           # nome esatto: vedi t.keys()
q16, q50, q84 = np.percentile(res, [16, 50, 84])
print(f"bias {q50:+.2f} mm  sigma_eff {0.5*(q84-q16):.2f} mm")
```

## Vertice 3D o altre coordinate

Nel YAML, sezione `labels`, elenca piu' target:

```yaml
labels:
  type: custom
  value:
    Vertex_x: Vertex_x
    Vertex_y: Vertex_y
    Vertex_z: Vertex_z
```

`get_model` legge `len(data_config.label_value)` e dimensiona da solo
l'uscita a 3 nodi; la MSE media sulle 3 componenti. Nessun'altra
modifica.

## Cambiare loss

In `networks/particle_net_regr.py`, `RegressionLoss` eredita da
`nn.MSELoss`. Per Huber o L1 basta cambiare la classe base
(`nn.HuberLoss`, `nn.L1Loss`).

## Perche' questo e' "canonico"

- backbone = `weaver.nn.model.ParticleNet.ParticleNet` (invariato);
- `FeatureConv`, `conv_params [(16,(64,64,64)),(16,(128,128,128)),
  (16,(256,256,256))]`, `fc_params [(256,0.1)]`: identici al
  `ParticleNetTagger` ufficiale;
- unica differenza dal tagger: `for_inference=False` (niente softmax),
  testa a `num_targets` nodi e loss MSE — le modifiche minime
  necessarie per passare da classificazione a regressione.
```
