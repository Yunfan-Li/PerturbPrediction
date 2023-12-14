import torch
import torch.nn.functional as F
from torch import nn


class ConditionalMLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_cond,
        dim_out=None,
        act_layer=None,
        norm_layer=None,
        condition_bias=1.0,
    ):
        super(ConditionalMLP, self).__init__()
        dim_out = dim_out or dim_in
        self.linear = nn.Linear(dim_in, dim_out)
        self.norm = (norm_layer or nn.LayerNorm)(dim_out)
        self.act = (act_layer or nn.ReLU)()

        self.cond_layers = nn.Sequential(self.act, nn.Linear(dim_cond, dim_out))
        self.cond_bias = condition_bias

    def forward(self, x, cond):
        x = self.linear(x)
        x = F.dropout(x, p=0.5, training=self.training)
        if cond is not None:
            cond = self.cond_layers(cond)
            x = x * (self.cond_bias + cond)
        x = self.act(x)
        return x


class ConditionalNet(nn.Module):
    def __init__(
        self,
        dim_feature=32,
        dim_cond_embed=32,
        dim_hidden=1024,
        dim_out=18211,
        n_blocks=3,
        skip_layers=(),
    ):
        super(ConditionalNet, self).__init__()

        self.skip_layers = skip_layers
        blocks = [
            ConditionalMLP(
                dim_in=dim_feature,
                dim_cond=dim_cond_embed,
                dim_out=dim_hidden,
                act_layer=nn.ReLU,
                norm_layer=nn.LayerNorm,
            )
        ]
        for i in range(1, n_blocks):
            blocks.append(
                ConditionalMLP(
                    dim_in=(dim_hidden + dim_feature)
                    if i in skip_layers
                    else dim_hidden,
                    dim_cond=dim_cond_embed,
                    dim_out=dim_hidden,
                    act_layer=nn.ReLU,
                    norm_layer=nn.BatchNorm1d,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(dim_hidden, dim_out)

        self.n_blocks = n_blocks
        self.timestep_embed_dim = dim_cond_embed

    def forward(self, x, condition):
        r = x
        for i in range(self.n_blocks):
            if i in self.skip_layers:
                x = torch.cat([x, r], dim=1)
            x = self.blocks[i](x, cond=condition)
        x_out = self.linear(x)
        return x_out


class Net(nn.Module):
    def __init__(
        self,
        gene_num,
        compound_num,
        sm_feature=None,
        type_rna=None,
        type_atac=None,
    ):
        super(Net, self).__init__()
        self.compound_num = compound_num
        self.gene_num = gene_num

        if sm_feature is None:
            self.sm_emb = nn.Embedding(self.gene_num, 64)
            self.sm_enc = None
        else:
            self.sm_emb = sm_feature
            self.sm_enc = nn.Sequential(
                nn.Linear(self.sm_emb.shape[1], 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 64),
            )
        self.type_rna = type_rna
        self.type_atac = type_atac
        self.atac_enc = nn.Sequential(
            nn.Linear(self.type_atac.shape[1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.gene_num, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )

        self.decoder = ConditionalNet(
            dim_feature=64,
            dim_cond_embed=128,
            dim_hidden=1024,
            dim_out=self.gene_num,
            n_blocks=4,
            skip_layers=(),
        )

    def forward(self, x, type, sm_name):
        if self.sm_enc is None:
            sm = self.sm_emb(sm_name)
        else:
            sm = self.sm_enc(self.sm_emb[sm_name])
        encode = self.encoder(x)
        atac = self.atac_enc(self.type_atac[type])

        # Set compound as the input and negative sample (+ cell type ATAC) as the condition
        x = sm
        cond = torch.cat([encode, atac], dim=1)
        pred = self.decoder(x, cond)

        return pred
