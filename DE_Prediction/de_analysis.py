import pandas as pd
import anndata as ad
import scanpy as sc
from dask import delayed
from dask.distributed import Client, LocalCluster
import os, binascii
import numpy as np
import limma_utils
import imp

imp.reload(limma_utils)

if __name__ == "__main__":
    bulk_adata_train = sc.read_h5ad("../processed_data/bulk_adata_tidy.h5ad")
    bulk_adata_test = sc.read_h5ad("./output/test_adata.h5ad")
    bulk_adata_test.obs.index = "test" + bulk_adata_test.obs.index
    print(bulk_adata_train.X.max(), bulk_adata_train.X.min(), bulk_adata_train.X.mean())
    print(bulk_adata_test.X.max(), bulk_adata_test.X.min(), bulk_adata_test.X.mean())
    data = np.concatenate((bulk_adata_train.X, bulk_adata_test.X.astype(float)), axis=0)
    meta = pd.concat([bulk_adata_train.obs, bulk_adata_test.obs], axis=0)
    bulk_adata = sc.AnnData(data, obs=meta)
    print(bulk_adata)

    de_pert_cols = [
        "sm_name",
        "sm_lincs_id",
        "SMILES",
        "dose_uM",
        "timepoint_hr",
        "cell_type",
    ]

    control_compound = "Dimethyl Sulfoxide"

    def _run_limma_for_cell_type(bulk_adata):
        import limma_utils

        bulk_adata = bulk_adata.copy()

        compound_name_col = de_pert_cols[0]

        # limma doesn't like dashes etc. in the compound names
        rpert_mapping = (
            bulk_adata.obs[compound_name_col]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .set_index(compound_name_col)["index"]
            .to_dict()
        )

        bulk_adata.obs["Rpert"] = bulk_adata.obs.apply(
            lambda row: rpert_mapping[row[compound_name_col]],
            axis="columns",
        ).astype("str")

        compound_name_to_Rpert = bulk_adata.obs.set_index(compound_name_col)[
            "Rpert"
        ].to_dict()
        ref_pert = compound_name_to_Rpert[control_compound]

        random_string = binascii.b2a_hex(os.urandom(15)).decode()

        limma_utils.limma_fit(
            bulk_adata,
            design="~0+Rpert+donor_id+plate_name+row",
            output_path=f"output/{random_string}_limma.rds",
            plot_output_path=f"output/{random_string}_voom",
            exec_path="limma_fit.r",
            verbose=False,
        )

        pert_de_dfs = []

        for pert in bulk_adata.obs["Rpert"].unique():
            if pert == ref_pert:
                continue

            pert_de_df = limma_utils.limma_contrast(
                fit_path=f"output/{random_string}_limma.rds",
                contrast="Rpert" + pert + "-Rpert" + ref_pert,
                exec_path="limma_contrast.r",
            )

            pert_de_df["Rpert"] = pert

            pert_obs = bulk_adata.obs[bulk_adata.obs["Rpert"].eq(pert)]
            for col in de_pert_cols:
                pert_de_df[col] = pert_obs[col].unique()[0]
            pert_de_dfs.append(pert_de_df)

        de_df = pd.concat(pert_de_dfs, axis=0)

        try:
            os.remove(f"output/{random_string}_limma.rds")
            os.remove(f"output/{random_string}_voom")
        except FileNotFoundError:
            pass

        return de_df

    run_limma_for_cell_type = delayed(_run_limma_for_cell_type)

    cluster = LocalCluster(
        n_workers=6,
        processes=True,
        threads_per_worker=1,
        memory_limit="20GB",
    )

    c = Client(cluster)

    cell_types = bulk_adata.obs["cell_type"].unique()
    de_dfs = []

    for cell_type in cell_types:
        cell_type_selection = bulk_adata.obs["cell_type"].eq(cell_type)
        cell_type_bulk_adata = bulk_adata[cell_type_selection].copy()

        de_df = run_limma_for_cell_type(cell_type_bulk_adata)

        de_dfs.append(de_df)

    de_dfs = c.compute(de_dfs, sync=True)
    de_df = pd.concat(de_dfs)

    def convert_de_df_to_anndata(de_df, pert_cols, de_sig_cutoff):
        de_df = de_df.copy()
        zero_pval_selection = de_df["P.Value"].eq(0)
        de_df.loc[zero_pval_selection, "P.Value"] = np.finfo(np.float64).eps

        de_df["sign_log10_pval"] = np.sign(de_df["logFC"]) * -np.log10(de_df["P.Value"])
        de_df["is_de"] = de_df["P.Value"].lt(de_sig_cutoff)
        de_df["is_de_adj"] = de_df["adj.P.Val"].lt(de_sig_cutoff)

        de_feature_dfs = {}
        for feature in [
            "is_de",
            "is_de_adj",
            "sign_log10_pval",
            "logFC",
            "P.Value",
            "adj.P.Val",
        ]:
            df = de_df.reset_index().pivot_table(
                index=["gene"],
                columns=pert_cols,
                values=[feature],
                dropna=True,
            )
            de_feature_dfs[feature] = df

        de_adata = ad.AnnData(de_feature_dfs["sign_log10_pval"].T, dtype=np.float64)
        de_adata.obs = de_adata.obs.reset_index()
        de_adata.obs = de_adata.obs.drop(columns=["level_0"])
        de_adata.obs.index = de_adata.obs.index.astype("string")

        de_adata.layers["is_de"] = de_feature_dfs["is_de"].to_numpy().T
        de_adata.layers["is_de_adj"] = de_feature_dfs["is_de_adj"].to_numpy().T
        de_adata.layers["logFC"] = de_feature_dfs["logFC"].to_numpy().T
        de_adata.layers["P.Value"] = de_feature_dfs["P.Value"].to_numpy().T
        de_adata.layers["adj.P.Val"] = de_feature_dfs["adj.P.Val"].to_numpy().T

        return de_adata

    de_adata = convert_de_df_to_anndata(de_df, de_pert_cols, 0.05)

    de_adata.obs.index = de_adata.obs.index.astype("str")

    sorting_index = de_adata.obs.sort_values(["sm_name", "cell_type"]).index
    de_adata = de_adata[sorting_index].copy()
    de_adata.obs
    sc.write("./output/de_adata.h5ad", de_adata)
