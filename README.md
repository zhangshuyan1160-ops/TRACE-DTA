# TRACE-DTA

TRACE-DTA predicts drug–target binding affinity from a ligand SMILES and one or more protein chains. The base predictor combines a molecular graph encoder with an ESM-2 protein encoder that uses sequence order and continuous residue contacts. A molecular relation encoder learns pairwise distances from a frozen SMILES language model. A target-conditioned adapter adds the resulting ligand representation to the base molecular representation before affinity prediction.

[Download the code](https://github.com/zhangshuyan1160-ops/TRACE-DTA/releases/download/v1.0.0/TRACE-DTA-code.zip) · [Download the study data and trained weights](https://github.com/zhangshuyan1160-ops/TRACE-DTA/releases/download/v1.0.0/TRACE-DTA-data.zip)

## Installation

Use Python 3.12. Run the commands below from the directory containing this README. Extract the companion data archive next to this directory so that `../TRACE-DTA-data/` contains `records/`, `splits/` and `weights/`.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

CPU and CUDA are supported. The commands below use CPU by default. Add `--device cuda` to feature extraction, training or prediction to use a CUDA-enabled PyTorch installation.

## Predict with the published Davis model

The companion data archive contains the weights used for the independent kinase screen. This example recovers Davis fold 1 from the authors’ input files and predicts its held-out pairs. The four input files can also be placed in `raw/davis/` using the URLs and SHA-256 digests in the data archive’s `sources.tsv`.

```sh
python -m trace_dta.prepare --data ../TRACE-DTA-data --raw raw/davis --fetch \
    --dataset davis --fold 1 --output work/davis
mkdir -p weights/esm2
curl -fL --retry 2 https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt \
    -o weights/esm2/esm2_t33_650M_UR50D.pt
curl -fL --retry 2 https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt \
    -o weights/esm2/esm2_t33_650M_UR50D-contact-regression.pt
python -m trace_dta.features protein --input work/davis/test.tsv --weights weights/esm2 \
    --output features/proteins
python -m trace_dta.predict --input work/davis/test.tsv --features features/proteins \
    --weights ../TRACE-DTA-data/weights/davis.safetensors --output work/davis_predictions.tsv
```

The inference input is a tab-separated file. Each row has a unique `sample`, a ligand `smiles` and complete protein `chains` separated by colons. Chain multiplicity is preserved. The output columns are `sample` and `predicted_pkd`. The molecular teacher is used during training.

To reproduce the kinase screening inputs, use `--dataset kinase_panel --output work/kinases` with the same Davis source directory. Run protein feature extraction and prediction on `work/kinases/pairs.tsv` with the Davis model. The 3,864 pKd predictions can be ranked within each assay and compared with the independent single-concentration inhibition measurements. The data archive identifies the original measurement table.

## Train on a study split

```sh
python -m trace_dta.features protein --input work/davis/proteins.tsv --weights weights/esm2 \
    --output features/proteins
python -m trace_dta.features teacher --input work/davis/pool.tsv --output work/davis_teacher.npz
python -m trace_dta.train --split work/davis --features features/proteins \
    --teacher work/davis_teacher.npz --seed 1 --output work/davis_trained.safetensors
```

1. Fit the base predictor to training pKd, standardized with the training mean and population standard deviation. The loss combines twice the Huber loss with a pairwise ordering term weighted by 0.1.
2. Freeze the base predictor and fit the molecular relation encoder on eligible molecules in the training pool. Match mean-normalized pairwise Euclidean distances between unit-length student and teacher embeddings. Teacher embeddings use the final hidden layer averaged over molecular tokens. The maximum teacher input length is 510 tokens including boundary tokens. Longer SMILES remain eligible for supervised affinity training.
3. Freeze both encoders and the base readout. Fit a target gate and a zero-initialized residual projection to the supervised affinity objective. Select the base predictor and the adapter by validation MSE. The unadapted predictor is an eligible starting point for adapter selection.

The supervised stages use AdamW with learning rate 0.001, weight decay 0.00001, gradient norm limit 5 and batch size 32. The default schedule has 80 validation intervals per supervised stage. Relation training uses up to 80 epochs or one hour. Its selection criterion is distance loss on a fixed set of up to 128 training molecules, including the initialized encoder as a candidate. A fresh fit follows this training procedure. For the reported screening scores, use the supplied trained weights.

Use `--dataset bindingdb` or `--dataset pdbbind` with the corresponding raw directory to prepare either benchmark. `--protocol sequence_cluster --fold 1` selects the first of three sequence-cluster folds. For low-label training, run, for example,

```sh
python -m trace_dta.prepare --data ../TRACE-DTA-data --raw raw/bindingdb --fetch \
    --dataset bindingdb --protocol label_budget --fold 1 --percent 10 --seed 1 \
    --output work/bindingdb_10
```

Here `train.tsv` contains 1,380 labelled pairs and `pool.tsv` contains all 10,746 outer-training input pairs. The subset seed selects the measured pairs. Set the training seed separately in `trace_dta.train`. Use `pool.tsv` for the molecular teacher. Supervised training holds the number of optimizer updates and validation events fixed across budgets.

## Pretrained representations and licence

ESM-2 uses `esm2_t33_650M_UR50D` with its contact-regression weights. The extractor streams the contact regression across attention layers for each complete chain. The optional protein extraction flag `--tile-long-chains` computes queries in blocks for chains longer than 2,500 residues and retains every residue pair. The molecular teacher is `seyonec/PubChem10M_SMILES_BPE_450k`, revision `c18fccd09b3326bf2d4633412c256d7db872156d`. For offline teacher extraction, pass a local snapshot of this revision with `--weights`. [ESM](https://github.com/facebookresearch/esm) · [Molecular teacher](https://huggingface.co/seyonec/PubChem10M_SMILES_BPE_450k/tree/c18fccd09b3326bf2d4633412c256d7db872156d)

TRACE-DTA source code is distributed under the MIT License. The contact implementation follows fair-esm, whose MIT notice is included. The companion data archive describes data sources, attribution and reuse terms. [License](LICENSE) · [Third-party notice](THIRD_PARTY_NOTICES.txt)
