import torch
import numpy as np
import scanpy as sc
import pandas as pd
from torch.utils.data import TensorDataset


def prepare_dataset_train_val(exclude_control=False, full_train=False):
    # Read train data
    de_train = pd.read_parquet("../data/de_train.parquet")

    # Exclude control data
    if exclude_control:
        de_train = de_train[de_train["control"] == False]

    # Read rna and atac baseline data
    print("Preparing RNA count...")
    multiome_train_rna_type = pd.read_parquet(
        "../processed_data/multiome_train_rna_type_count.parquet"
    )
    multiome_types = multiome_train_rna_type.iloc[:, 0].values
    adata = sc.AnnData(multiome_train_rna_type.iloc[:, 1:].values)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    normalized_rna_count = adata.X
    print("Done.")

    # Convert cell type to integer
    print("Computing cell type count...")
    train_cell_type = de_train["cell_type"].values
    train_cell_type_int = (
        de_train["cell_type"].rank(method="dense").astype(int).values - 1
    )
    cell_type_map = dict()
    for k in range(train_cell_type_int.max() + 1):
        cell_type_map[k] = train_cell_type[train_cell_type_int == k][0]

    # Prepare cell type count
    cell_type_rna_count = np.zeros_like(normalized_rna_count)
    for k, v in cell_type_map.items():
        cell_type_idx = np.where(multiome_types == v)[0][0]
        cell_type_rna_count[k] = normalized_rna_count[cell_type_idx]
    cell_type_rna_count = torch.tensor(cell_type_rna_count, dtype=torch.float)
    print("Done.")

    # Convert sm_name to integer
    print("Loading compound feature...")
    train_sm_name = de_train["sm_name"].values
    train_sm_name_int = de_train["sm_name"].rank(method="dense").astype(int).values - 1
    sm_name_map = dict()
    for k in range(train_sm_name_int.max() + 1):
        sm_name_map[k] = train_sm_name[train_sm_name_int == k][0]

    # Prepare sm feature from pre-trained model
    sm_data = pd.read_parquet("../processed_data/sm_feature.parquet")
    sm_data_name = sm_data.iloc[:, 0].values
    sm_data_feature = sm_data.iloc[:, 2:].values
    sm_feature = np.zeros_like(sm_data_feature)
    for k, v in sm_name_map.items():
        sm_name_index = np.where(sm_data_name == v)[0][0]
        sm_feature[k] = sm_data_feature[sm_name_index]
    sm_feature = torch.tensor(sm_feature, dtype=torch.float)
    sm_feature = sm_feature / sm_feature.norm(dim=1, keepdim=True)
    if exclude_control:
        sm_feature = sm_feature[:-2]
    print("Done.")

    # Prepare train de data
    train_de = de_train.drop(
        ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"], axis=1
    ).values

    # Select some perturbation of B and Myeloid cells for validation (Other choices are possible)
    val_type_mask = (de_train.loc[:, "cell_type"] == "B cells") | (
        de_train.loc[:, "cell_type"] == "Myeloid cells"
    )
    val_sm_mask = (
        (de_train.loc[:, "sm_name"] == "Idelalisib")
        | (de_train.loc[:, "sm_name"] == "Linagliptin")
        | (de_train.loc[:, "sm_name"] == "Alvocidib")
        | (de_train.loc[:, "sm_name"] == "R428")
        | (de_train.loc[:, "sm_name"] == "Foretinib")
        | (de_train.loc[:, "sm_name"] == "Penfluridol")
        | (de_train.loc[:, "sm_name"] == "O-Demethylated Adapalene")
        | (de_train.loc[:, "sm_name"] == "CHIR-99021")
    )
    val_mask = (val_type_mask & val_sm_mask).values
    val_de = train_de[val_mask]
    val_cell_type_int = train_cell_type_int[val_mask]
    val_sm_name_int = train_sm_name_int[val_mask]
    if not full_train:
        train_de = train_de[~val_mask]
        train_cell_type_int = train_cell_type_int[~val_mask]
        train_sm_name_int = train_sm_name_int[~val_mask]

    # Prepare dataset
    train_dataset = TensorDataset(
        torch.tensor(train_de, dtype=torch.float),
        torch.tensor(train_cell_type_int, dtype=torch.long),
        torch.tensor(train_sm_name_int, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_de, dtype=torch.float),
        torch.tensor(val_cell_type_int, dtype=torch.long),
        torch.tensor(val_sm_name_int, dtype=torch.long),
    )

    return (
        train_dataset,
        val_dataset,
        cell_type_rna_count,
        sm_feature,
    )


def prepare_dataset_test():
    # Read train and test meta data
    de_train = pd.read_parquet("../data/de_train.parquet")
    de_train = de_train[de_train["control"] == False]
    id_map = pd.read_csv("../data/id_map.csv", index_col=0)
    print("Computing cell type...")
    train_cell_type = de_train["cell_type"].values
    train_cell_type_int = (
        de_train["cell_type"].rank(method="dense").astype(int).values - 1
    )
    cell_type_map = dict()
    for k in range(train_cell_type_int.max() + 1):
        cell_type_map[k] = train_cell_type[train_cell_type_int == k][0]
    test_cell_type = id_map["cell_type"].values
    test_cell_type_int = test_cell_type
    for k, v in cell_type_map.items():
        test_cell_type_int[test_cell_type == v] = k
    test_cell_type_int = test_cell_type_int.astype(int)
    print("Done.")

    print("Compute compound...")
    train_sm_name = de_train["sm_name"].values
    train_sm_name_int = de_train["sm_name"].rank(method="dense").astype(int).values - 1
    sm_name_map = dict()
    for k in range(train_sm_name_int.max() + 1):
        sm_name_map[k] = train_sm_name[train_sm_name_int == k][0]
    test_sm_name = id_map["sm_name"].values
    test_sm_name_int = test_sm_name
    for k, v in sm_name_map.items():
        test_sm_name_int[test_sm_name == v] = k
    test_sm_name_int = test_sm_name_int.astype(int)
    print("Done.")

    test_dataset = TensorDataset(
        torch.tensor(test_cell_type_int, dtype=torch.long),
        torch.tensor(test_sm_name_int, dtype=torch.long),
    )

    return test_dataset
