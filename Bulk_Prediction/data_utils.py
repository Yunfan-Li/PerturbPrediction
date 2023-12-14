import torch
import numpy as np
import scanpy as sc
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.feature_extraction.text import TfidfTransformer


def prepare_dataset_train_val(exclude_positive=False, full_train=False):
    # Read bulk data
    bulk_adata = sc.read_h5ad("../processed_data/bulk_adata.h5ad")

    negative_sm = "Dimethyl Sulfoxide"
    positive_sm_1 = "Belinostat"
    positive_sm_2 = "Dabrafenib"

    # Exclude control data
    if exclude_positive:
        bulk_adata = bulk_adata[bulk_adata.obs["sm_name"] != positive_sm_1]
        bulk_adata = bulk_adata[bulk_adata.obs["sm_name"] != positive_sm_2]
    negative_bulk_adata = bulk_adata[bulk_adata.obs["sm_name"] == negative_sm].copy()
    bulk_adata = bulk_adata[bulk_adata.obs["sm_name"] != negative_sm].copy()

    # Transfer compound name to int
    sm_names = sorted(bulk_adata.obs["sm_name"].unique())
    sm_name_map = dict()
    for k in range(len(sm_names)):
        sm_name_map[sm_names[k]] = k

    print("Preprocessing negative bulk data...")
    negative_bulk_adata_rc = negative_bulk_adata.X.copy()
    sc.pp.normalize_total(negative_bulk_adata, target_sum=1e4)
    negative_bulk_adata_sf = np.array(
        (negative_bulk_adata_rc.sum(axis=1) / 1e4).tolist()
    ).reshape(-1, 1)
    sc.pp.log1p(negative_bulk_adata)
    sc.pp.scale(negative_bulk_adata)
    negative_bulk_mean = torch.from_numpy(negative_bulk_adata.var["mean"].values)
    negative_bulk_std = torch.from_numpy(negative_bulk_adata.var["std"].values)
    print("Done.")

    bulk_adata_rc = bulk_adata.X.copy()

    print("Preparing RNA and ATAC...")
    de_train = pd.read_parquet("../data/de_train.parquet")
    train_cell_type_int = (
        de_train["cell_type"].rank(method="dense").astype(int).values - 1
    )
    multiome_train_rna_type = pd.read_parquet(
        "../processed_data/multiome_train_rna_type_count.parquet"
    )
    multiome_types = multiome_train_rna_type.iloc[:, 0].values
    adata = sc.AnnData(multiome_train_rna_type.iloc[:, 1:].values)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    normalized_rna = adata.X
    multiome_train_atac_type = pd.read_parquet(
        "../processed_data/multiome_train_atac_type_count.parquet"
    )
    adata = sc.AnnData(multiome_train_atac_type.iloc[:, 1:].values)
    tfidf = TfidfTransformer()
    adata.X = tfidf.fit_transform(adata.X).toarray()
    sc.pp.scale(adata)
    normalized_atac = adata.X

    cell_type_map = dict()
    train_cell_type = de_train["cell_type"].values
    for k in range(train_cell_type_int.max() + 1):
        cell_type_map[train_cell_type[train_cell_type_int == k][0]] = k
    cell_type_rna = np.zeros_like(normalized_rna)
    cell_type_atac = np.zeros_like(normalized_atac)
    for k, v in cell_type_map.items():
        cell_type_idx = np.where(multiome_types == k)[0][0]
        cell_type_rna[v] = normalized_rna[cell_type_idx]
        cell_type_atac[v] = normalized_atac[cell_type_idx]
    cell_type_rna = torch.tensor(cell_type_rna, dtype=torch.float)
    cell_type_atac = torch.tensor(cell_type_atac, dtype=torch.float)

    # Prepare sm feature
    print("Loading sm_feature...")
    sm_data = pd.read_parquet("../processed_data/smiles_count.parquet")
    sm_data_name = sm_data.iloc[:, 0].values
    sm_data_feature = sm_data.iloc[:, 2:].values
    sm_feature = np.zeros_like(sm_data_feature)
    for k, v in sm_name_map.items():
        sm_name_index = np.where(sm_data_name == k)[0][0]
        sm_feature[v] = sm_data_feature[sm_name_index]
    tfidf = TfidfTransformer()
    sm_feature = tfidf.fit_transform(sm_feature).toarray()
    sm_feature = torch.tensor(sm_feature, dtype=torch.float)
    print("Done.")

    # Prepare train data
    print("Preparing train data...")
    bulk_number = bulk_adata.X.shape[0]
    negative_x = torch.zeros((bulk_number, 18211), dtype=torch.float)
    negative_sf = torch.zeros((bulk_number, 1), dtype=torch.float)
    cell_types_int = torch.zeros((bulk_number), dtype=torch.long)
    sm_names_int = torch.zeros((bulk_number), dtype=torch.long)
    target_rc = torch.from_numpy(bulk_adata_rc).float()
    for i in range(bulk_number):
        negative_index = (
            (negative_bulk_adata.obs["donor_id"] == bulk_adata.obs["donor_id"].iloc[i])
            & (
                negative_bulk_adata.obs["plate_name"]
                == bulk_adata.obs["plate_name"].iloc[i]
            )
            & (
                negative_bulk_adata.obs["cell_type"]
                == bulk_adata.obs["cell_type"].iloc[i]
            )
            & (negative_bulk_adata.obs["row"] == bulk_adata.obs["row"].iloc[i])
        )
        negative_x[i] = torch.from_numpy(negative_bulk_adata.X[negative_index])
        negative_sf[i] = torch.from_numpy(negative_bulk_adata_sf[negative_index])
        cell_types_int[i] = cell_type_map[bulk_adata.obs["cell_type"].iloc[i]]
        sm_names_int[i] = sm_name_map[bulk_adata.obs["sm_name"].iloc[i]]

    # Train Val Split (other splits are acceptable)
    val_type_mask = bulk_adata.obs["cell_type"] == "NK cells"
    val_sm_mask = ~bulk_adata.obs["sm_name"].isin(
        (
            "Belinostat",
            "Dabrafenib",
            "Alvocidib",
            "CHIR-99021",
            "Crizotinib",
            "Dactolisib",
            "Foretinib",
            "Idelalisib",
            "LDN 193189",
            "Linagliptin",
            "MLN 2238",
            "O-Demethylated Adapalene",
            "Oprozomib (ONX 0912)",
            "Palbociclib",
            "Penfluridol",
            "Porcn Inhibitor III",
            "R428",
        )
    )
    val_mask = (val_type_mask & val_sm_mask).values
    val_index = np.where(val_mask)[0]
    train_index = np.array(list(set(range(bulk_number)) - set(val_index)))
    negative_x_val = negative_x[val_index]
    negative_sf_val = negative_sf[val_index]
    cell_types_int_val = cell_types_int[val_index]
    sm_names_int_val = sm_names_int[val_index]
    target_rc_val = target_rc[val_index]
    if not full_train:
        negative_x = negative_x[train_index]
        negative_sf = negative_sf[train_index]
        cell_types_int = cell_types_int[train_index]
        sm_names_int = sm_names_int[train_index]
        target_rc = target_rc[train_index]

    train_dataset = TensorDataset(
        negative_x, negative_sf, cell_types_int, sm_names_int, target_rc
    )
    val_dataset = TensorDataset(
        negative_x_val,
        negative_sf_val,
        cell_types_int_val,
        sm_names_int_val,
        target_rc_val,
    )

    return (
        train_dataset,
        val_dataset,
        sm_feature,
        cell_type_rna,
        cell_type_atac,
        negative_bulk_mean,
        negative_bulk_std,
    )


def prepare_dataset_test():
    bulk_adata = sc.read_h5ad("../processed_data/bulk_adata.h5ad")
    negative_sm = "Dimethyl Sulfoxide"

    negative_bulk_adata = bulk_adata[bulk_adata.obs["sm_name"] == negative_sm].copy()
    bulk_adata = bulk_adata[bulk_adata.obs["sm_name"] != negative_sm].copy()
    train_meta = bulk_adata.obs

    sm_names = sorted(bulk_adata.obs["sm_name"].unique())
    sm_name_map = dict()
    for k in range(len(sm_names)):
        sm_name_map[sm_names[k]] = k

    print("Preprocessing negative bulk data...")
    negative_bulk_adata_rc = negative_bulk_adata.X.copy()
    sc.pp.normalize_total(negative_bulk_adata, target_sum=1e4)
    negative_bulk_adata_sf = np.array(
        (negative_bulk_adata_rc.sum(axis=1) / 1e4).tolist()
    ).reshape(-1, 1)
    sc.pp.log1p(negative_bulk_adata)
    sc.pp.scale(negative_bulk_adata)
    negative_bulk_mean = torch.from_numpy(negative_bulk_adata.var["mean"].values)
    negative_bulk_std = torch.from_numpy(negative_bulk_adata.var["std"].values)
    print("Done.")

    print("Preparing RNA and ATAC...")
    de_train = pd.read_parquet("../data/de_train.parquet")
    train_cell_type_int = (
        de_train["cell_type"].rank(method="dense").astype(int).values - 1
    )
    multiome_train_rna_type = pd.read_parquet(
        "../processed_data/multiome_train_rna_type_count.parquet"
    )
    multiome_types = multiome_train_rna_type.iloc[:, 0].values
    adata = sc.AnnData(multiome_train_rna_type.iloc[:, 1:].values)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    normalized_rna = adata.X
    multiome_train_atac_type = pd.read_parquet(
        "../processed_data/multiome_train_atac_type_count.parquet"
    )
    adata = sc.AnnData(multiome_train_atac_type.iloc[:, 1:].values)
    tfidf = TfidfTransformer()
    adata.X = tfidf.fit_transform(adata.X).toarray()
    sc.pp.scale(adata)
    normalized_atac = adata.X
    cell_type_map = dict()
    train_cell_type = de_train["cell_type"].values
    for k in range(train_cell_type_int.max() + 1):
        cell_type_map[train_cell_type[train_cell_type_int == k][0]] = k
    cell_type_rna = np.zeros_like(normalized_rna)
    cell_type_atac = np.zeros_like(normalized_atac)
    for k, v in cell_type_map.items():
        cell_type_idx = np.where(multiome_types == k)[0][0]
        cell_type_rna[v] = normalized_rna[cell_type_idx]
        cell_type_atac[v] = normalized_atac[cell_type_idx]
    cell_type_rna = torch.tensor(cell_type_rna, dtype=torch.float)
    cell_type_atac = torch.tensor(cell_type_atac, dtype=torch.float)

    # Prepare sm feature
    print("Loading sm_feature...")
    sm_data = pd.read_parquet("../processed_data/smiles_count.parquet")
    sm_data_name = sm_data.iloc[:, 0].values
    sm_data_feature = sm_data.iloc[:, 2:].values
    sm_feature = np.zeros_like(sm_data_feature)
    for k, v in sm_name_map.items():
        sm_name_index = np.where(sm_data_name == k)[0][0]
        sm_feature[v] = sm_data_feature[sm_name_index]
    tfidf = TfidfTransformer()
    sm_feature = tfidf.fit_transform(sm_feature).toarray()
    sm_feature = torch.tensor(sm_feature, dtype=torch.float)
    # sm_feature = sm_feature / sm_feature.norm(dim=1, keepdim=True)
    print("Done.")

    # Prepare train data
    print("Preparing train data...")
    negative_x = torch.zeros((765, 18211), dtype=torch.float)
    negative_sf = torch.zeros((765, 18211), dtype=torch.float)
    cell_types_int = torch.zeros((765), dtype=torch.long)
    sm_names_int = torch.zeros((765), dtype=torch.long)
    test_meta = pd.read_csv("processed_data/test_meta.csv")
    # select common columns between train_meta and test_meta
    train_meta = train_meta.loc[:, test_meta.columns]
    for i in range(765):
        negative_index = (
            (negative_bulk_adata.obs["donor_id"] == test_meta["donor_id"].iloc[i])
            & (negative_bulk_adata.obs["plate_name"] == test_meta["plate_name"].iloc[i])
            & (negative_bulk_adata.obs["cell_type"] == test_meta["cell_type"].iloc[i])
            & (negative_bulk_adata.obs["row"] == test_meta["row"].iloc[i])
        )
        negative_x[i] = torch.from_numpy(negative_bulk_adata.X[negative_index])
        negative_sf[i] = torch.from_numpy(negative_bulk_adata_sf[negative_index])
        cell_types_int[i] = cell_type_map[test_meta["cell_type"].iloc[i]]
        sm_names_int[i] = sm_name_map[test_meta["sm_name"].iloc[i]]

    test_dataset = TensorDataset(negative_x, negative_sf, cell_types_int, sm_names_int)

    return (
        test_dataset,
        sm_feature,
        cell_type_rna,
        cell_type_atac,
        train_meta,
        test_meta,
        negative_bulk_mean,
        negative_bulk_std,
    )
