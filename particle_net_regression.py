"""
Model config per weaver-core: ParticleNet UFFICIALE come backbone,
adattato a REGRESSIONE del vertice (MSE).

Importa il ParticleNet canonico da weaver
(weaver.nn.model.ParticleNet). Non riscrive l'architettura: usa la
stessa FeatureConv + EdgeConv + pooling del tagger di Huilin Qu, e
sostituisce solo la testa fully-connected finale con un'uscita di
regressione a `num_targets` nodi (default 1 = Vertex_z).

Uso con weaver:
  weaver --data-config data/crilin_vertex.yaml \
         --network-config networks/particle_net_regr.py \
         ...

`get_model` e `get_loss` sono le due funzioni che weaver richiede
in un network-config.
"""
import torch
import torch.nn as nn

# ParticleNet ufficiale del framework weaver-core
from weaver.nn.model.ParticleNet import ParticleNet, FeatureConv


class ParticleNetRegressor(nn.Module):
    """
    Backbone ParticleNet ufficiale + testa lineare di regressione.
    Ricalca ParticleNetTagger1Path ma con un solo "path" (i PF hit)
    e output continuo invece dei logit di classe.
    """

    def __init__(self, pf_features_dims, num_targets,
                 conv_params, fc_params,
                 use_fusion=False, use_fts_bn=True,
                 use_counts=True, for_inference=False, **kwargs):
        super().__init__()
        # proiezione delle feature per-hit (come nel tagger ufficiale)
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        # ParticleNet ufficiale: num_classes qui e' la dimensione di
        # output della testa MLP interna -> la usiamo come num_targets.
        self.pn = ParticleNet(
            input_dims=32,
            num_classes=num_targets,
            conv_params=conv_params,
            fc_params=fc_params,
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            for_inference=False,   # niente softmax: e' regressione
        )

    def forward(self, points, features, mask):
        # firma identica al ParticleNetTagger ufficiale
        return self.pn(points, self.pf_conv(features * mask) * mask, mask)


def get_model(data_config, **kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ]
    fc_params = [(256, 0.1)]

    pf_features_dims = len(data_config.input_dicts["pf_features"])
    num_targets = len(data_config.label_value)  # = 1 per il solo Vertex_z

    model = ParticleNetRegressor(
        pf_features_dims=pf_features_dims,
        num_targets=num_targets,
        conv_params=conv_params,
        fc_params=fc_params,
        use_fusion=False,
        **kwargs,
    )

    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()
        },
        "output_names": ["output"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_hits"} for k in data_config.input_names},
            "output": {0: "N"},
        },
    }
    return model, model_info


class RegressionLoss(nn.MSELoss):
    """MSE. Sostituibile con nn.HuberLoss / nn.L1Loss se preferisci."""
    pass


def get_loss(data_config, **kwargs):
    return RegressionLoss()
