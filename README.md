# CRILIN — regressione con ParticleNet per sciami di elettroni

Point cloud degli hit del calorimetro CRILIN (matrice 7×7×5 di cristalli PbF₂)
→ rete ParticleNet (EdgeConv) → regressione di uno scalare per evento con MSE.

Il codice prende spunto dal training usato in [arXiv:2606.05111](https://arxiv.org/abs/2606.05111) per la
compensazione software degli sciami adronici ma è adattato al task degli
elettroni: ricostruire una **posizione** dalla **forma** dello sciame, non
un'energia dalla sua scala.

## Obiettivo e stato del target

L'obiettivo finale è regredire il **punto di inizio dello sciame** (primo
vertice di interazione hard) dalla forma 3D dello sciame. **Attenzione**:  
la pipeline si valida
sulle posizioni la cui verità **è** disponibile: `Vertex_x`/`Vertex_y` (spot
del fascio, ±2.5 mm) — lo stesso tipo di task (geometria dello sciame →
posizione), solo sull'asse trasverso. In alternativa `--target PrimaryEnergy`
riproduce il task del paper. 

## Variabili di input

Per ogni cella (max 245 = 7×7×5),  la frazione di
energia:

| variabile | significato |
|---|---|
| `Hit_x`, `Hit_y`, `Hit_z` | posizione del centro cella [mm] |
| `E` smearata | `Poisson(pe·E)/pe` in GeV, soglia 50 MeV (fotostatistica; `--pe`, default 0.2 pe/MeV  — la resa Cherenkov misurata su questi ntuple è ~24.4 pe/MeV) |
| `E/E_sum` | frazione di energia della cella: rende esplicita la **forma** dello sciame, indipendente dalla scala di energia |

Fisicamente: lo spostamento del punto di inizio sciame trasla rigidamente il
profilo longitudinale (il baricentro ⟨z⟩ è l'osservabile singola più sensibile)
e a parità di profondità uno sciame più "giovane" è più stretto trasversalmente.
Una posizione vive nella distribuzione relativa dell'energia tra le celle (da
qui la frazione), mentre l'energia assoluta serve da contesto perché la
profondità dello sciame cresce come ln(E); l'EdgeConv impara queste
correlazioni direttamente dalla point cloud.

## Dati

Su EOS: `/eos/user/m/mcampana/PNet/`
- `uniform_2.5mmsquared_30_100GeV_4mmAldesign_cat.root` — 200k eventi, spettro uniforme 30–100 GeV → **training**
- `fixed99GeV_2.5mmsquared_30_100GeV_4mmAldesign.root` — 99 GeV fissi → **test indipendente**

Tree `events`; unità: energie in MeV, posizioni in mm.

**Convenzione dell'asse z (invertita)**: z grande = lato fascio. Il primario è
generato a z = +400 mm, la faccia d'ingresso del calorimetro è il layer a
z = +92.5 mm e lo sciame si sviluppa verso z decrescenti (layer più profondo a
z = −95.5 mm). La profondità è quindi (z_fronte − z).

## Ambiente (lxplus con GPU)

Nessuna installazione, basta lo stack LCG:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh
```

(TensorFlow 2.21 + CUDA, uproot, awkward, matplotlib.)

## Uso

```bash
# training sul campione uniforme (default: --target Vertex_x, checkpoint sul best val_loss)
python electron.py --file /eos/user/m/mcampana/PNet/uniform_2.5mmsquared_30_100GeV_4mmAldesign_cat.root \
                   --train --epochs 60

# predizione sul campione a 99 GeV con i pesi salvati
python electron.py --file /eos/user/m/mcampana/PNet/uniform_2.5mmsquared_30_100GeV_4mmAldesign_cat.root \
                   --predict-file /eos/user/m/mcampana/PNet/fixed99GeV_2.5mmsquared_30_100GeV_4mmAldesign.root \
                   --output prediction_99gev.root

# valutazione: residui, bias, sigma_eff (16-84), plot
python evaluate.py prediction_99gev.root --outdir plots
```

Test rapido (pochi eventi/epoche): aggiungere `--max-events 5000 --epochs 2`.

`prediction.root` contiene, allineati per evento: `EventID`, `PrimaryEnergy`
[GeV], `VD_energy` [GeV], il target vero, `pred_<target>` e il flag `selected`
(la selezione `E_sum > 0.5 GeV`; per gli eventi scartati la predizione è 0,
come in pion.py) — formato compatibile con gli script di
[offline-compensation-crilin-analysis](https://github.com/raeubaen/offline-compensation-crilin-analysis).

## File

| file | scopo |
|---|---|
| `electron.py` | training + predizione (adattamento di pion.py) |
| `evaluate.py` | residui, bias, σ_eff, plot da `prediction.root` |
