import scanpy as sc
import pandas as pd
import numpy as np


# Mean Rowwise Root Mean Squared Error
def MRRMSE(y_true, y_pred):
    return np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)))


sample_submission = pd.read_csv("../data/sample_submission.csv", index_col=0, header=0)
meta = pd.read_csv("../data/id_map.csv")
de_adata = sc.read_h5ad("./output/de_adata.h5ad")
de_train = pd.read_parquet("../data/de_train.parquet")
de_train_BM = de_train[
    (de_train["cell_type"] == "B cells") | (de_train["cell_type"] == "Myeloid cells")
].copy()
de_train_BM.index = range(len(de_train_BM))
de_train_others = de_train[
    (de_train["cell_type"] != "B cells") & (de_train["cell_type"] != "Myeloid cells")
].copy()
de_train_others.index = range(len(de_train_others))

de_train_pred = np.zeros((34, 18211), dtype=float)
for i in range(34):
    index = (de_adata.obs["sm_name"] == de_train_BM["sm_name"][i]) & (
        de_adata.obs["cell_type"] == de_train_BM["cell_type"][i]
    )
    de_train_pred[i] = de_adata.X[index]
print(
    MRRMSE(
        de_train_pred[de_train_BM["cell_type"] == "B cells"],
        de_train_BM[de_train_BM["cell_type"] == "B cells"].iloc[:, -18211:].values,
    )
)
print(
    de_train_pred[de_train_BM["cell_type"] == "B cells"].mean(),
    de_train_BM[de_train_BM["cell_type"] == "B cells"].iloc[:, -18211:].values.mean(),
)
print(
    MRRMSE(
        de_train_pred[de_train_BM["cell_type"] == "Myeloid cells"],
        de_train_BM[de_train_BM["cell_type"] == "Myeloid cells"]
        .iloc[:, -18211:]
        .values,
    )
)
print(
    de_train_pred[de_train_BM["cell_type"] == "Myeloid cells"].mean(),
    de_train_BM[de_train_BM["cell_type"] == "Myeloid cells"]
    .iloc[:, -18211:]
    .values.mean(),
)
print(MRRMSE(de_train_pred, de_train_BM.iloc[:, -18211:].values))
print(de_train_pred.mean(), de_train_BM.iloc[:, -18211:].values.mean())

de_train_pred = np.zeros((580, 18211), dtype=float)
for i in range(580):
    index = (de_adata.obs["sm_name"] == de_train_others["sm_name"][i]) & (
        de_adata.obs["cell_type"] == de_train_others["cell_type"][i]
    )
    de_train_pred[i] = de_adata.X[index]
print(MRRMSE(de_train_pred, de_train_others.iloc[:, -18211:].values))
print(de_train_pred.mean(), de_train_others.iloc[:, -18211:].values.mean())

de_pred = np.zeros((255, 18211), dtype=float)
for i in range(255):
    index = (de_adata.obs["sm_name"] == meta["sm_name"][i]) & (
        de_adata.obs["cell_type"] == meta["cell_type"][i]
    )
    de_pred[i] = de_adata.X[index]
submit_submission = pd.DataFrame(
    de_pred,
    index=sample_submission.index,
    columns=sample_submission.columns,
)
print(submit_submission)
submit_submission.to_csv("submission.csv")
