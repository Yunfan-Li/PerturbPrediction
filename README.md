# Single-Cell Perturbation Prediction

Code for the NeurIPS 2023 Competition "Open Problems – Single-Cell Perturbation Prediction" (<https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview>). The two folders correspond to the two prediction pipelines.

## Installation

```bash
git clone https://github.com/Yunfan-Li/PerturbPrediction.git
cd PerturbPrediction
conda env create -f environment.yml
conda activate PerturbPrediction
```

Note: the code support multi-gpu training.

Please refer to the original repository (<https://github.com/openproblems-bio/neurips-2023-scripts>) to prepare the environment for Limma analysis. The (preprocessed) data could be downloaded from the google drive (<https://drive.google.com/drive/folders/1XQUM8izC7QsUsI9i3uRzPMbDLCnuRsJ6?usp=sharing>).

## DE value prediction

To train the model, simply run the following command:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 main.py 
```

Here the `--nproc_per_node` corresponds to the number of GPU used for training, please modify the `--batch_size` param accordingly if you change its value. When the `--full_train` param is not enabled, the code would select a portion of training data as validation set according to the criterion set in the `data_utils.py` file, else the model would be trained on all data. Additionally, the `--pre_type` and `--pre_sm` params would use the average RNA count and compound features from pre-trained models ChemBERT as cell type and compound representation, respectively. Otherwise, the cell type and compound representation would be jointly learned with the model.

After training (with the `--full_train` param), simply run the following command to make the prediction:

```bash
python inference.py
```

Note that the `--pre_type` and `--pre_sm` params need to be consistent with the training.

## Bulk value prediction

To train the model, simply run the following command:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 main.py 
```

After training (with the `--full_train` param), simply run the following command to make the prediction:

```bash
python inference.py
```

The params are similar with the aboves ones in the DE value prediction paradigm.

Next, please run the following script to run the Limma analyis (remember to set your R path in the `limma_utils.py` file):

```bash
python de_analysis.py
```

To transform the Limma result to submission file, please run:

```bash
python format.py
```
