import torch
from torch import nn


class Net(nn.Module):
    def __init__(
        self,
        type_num,
        compound_num,
        gene_num,
        cell_type_rna_count=None,
        sm_feature=None,
    ):
        super(Net, self).__init__()
        self.type_num = type_num
        self.compound_num = compound_num
        self.gene_num = gene_num

        if cell_type_rna_count is None:
            self.type_emb = nn.Embedding(6, 64)
            self.type_enc = None
        else:
            self.type_emb = cell_type_rna_count
            self.type_enc = nn.Sequential(
                nn.Linear(self.type_emb.shape[1], 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
        if sm_feature is None:
            self.sm_emb = nn.Embedding(144, 64)
            self.sm_enc = None
        else:
            self.sm_emb = sm_feature
            self.sm_enc = nn.Sequential(
                nn.Linear(self.sm_emb.shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

        self.predictor = nn.Sequential(
            nn.Linear(128, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, self.gene_num),
        )

    def forward(self, cell_type, sm_name):
        if self.type_enc is None:
            type = self.type_emb(cell_type)
        else:
            type = self.type_enc(self.type_emb[cell_type])
        if self.sm_enc is None:
            sm = self.sm_emb(sm_name)
        else:
            sm = self.sm_enc(self.sm_emb[sm_name])
        embedding = torch.cat([type, sm], dim=1)
        return self.predictor(embedding)
