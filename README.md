# GraFITi

This is the source code for the paper [GraFITi: Graphs for Forecasting of Irregularly sampled Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/29560) published in AAAI 2024


# Requirements
python                    3.8.11

Pytorch                   1.9.0

sklearn                   0.0

numpy                     1.19.3

pandas                    1.5

# Training and Evaluation

We provided the script to run all the datasets with hyperparameters for fold ``0``. With these scripts, one can reproduce the results.

```
train_grafiti.py --epochs 200 --learn-rate 0.001 --batch-size 128 --attn-head 4 --latent-dim 64 --nlayers 4 --dataset ushcn --fold 0 -ct 36 -ft 0
train_grafiti.py --epochs 200 --learn-rate 0.001 --batch-size 128 --attn-head 4 --latent-dim 64 --nlayers 2 --dataset physionet2012 --fold 0 -ct 36 -ft 0
train_grafiti.py --epochs 200 --learn-rate 0.001 --batch-size 64 --attn-head 4 --latent-dim 128 --nlayers 1 --dataset mimiciii --fold 0 -ct 36 -ft 0
train_grafiti.py --epochs 200 --learn-rate 0.001 --batch-size 128 --attn-head 1 --latent-dim 128 --nlayers 1 --dataset mimiciv --fold 0 -ct 36 -ft 0
```

MIMIC-IV and MIMIC-III require permissions to download the data. Once, the datasets are downloaded, you can add them to the folder .tsdm/rawdata/ and use the TSDM package to extract the folds. We use TSDM package provided by Scholz .et .al from [https://openreview.net/forum?id=a-bD9-0ycs0]


# Edit: Extension for more experiments
Recently more and more IMTS forecasting works have been using the evaluation protocol as proposed by Zhang et al in "Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach" (ICML 2024). Hence, we updated our code so future works can be compared to GraFITi easily. We apply the same hyperparameter search in our original paper and yield these results:

| Algorithm       | PhysioNet MSE (×10⁻³) | PhysioNet MAE (×10⁻²) | MIMIC MSE (×10⁻²) | MIMIC MAE (×10⁻²) | Human Activity MSE (×10⁻³) | Human Activity MAE (×10⁻²) | USHCN MSE (×10⁻¹) | USHCN MAE (×10⁻¹) |
|----------------|-----------------------|------------------------|-------------------|------------------|-----------------------------|----------------------------|-------------------|------------------|
| DLinear        | 41.86 ± 0.05          | 15.52 ± 0.03           | 4.90 ± 0.00       | 16.29 ± 0.05     | 4.03 ± 0.01                 | 4.21 ± 0.01                | 6.21 ± 0.00       | 3.88 ± 0.02      |
| TimesNet       | 16.48 ± 0.11          | 6.14 ± 0.03            | 5.88 ± 0.08       | 13.62 ± 0.07     | 3.12 ± 0.01                 | 3.56 ± 0.02                | 5.58 ± 0.05       | 3.60 ± 0.04      |
| PatchTST       | 12.00 ± 0.23          | 6.02 ± 0.14            | 3.78 ± 0.03       | 12.43 ± 0.10     | 4.29 ± 0.14                 | 4.80 ± 0.09                | 5.75 ± 0.01       | 3.57 ± 0.02      |
| Crossformer    | 6.66 ± 0.11           | 4.81 ± 0.11            | 2.65 ± 0.10       | 9.56 ± 0.29      | 4.29 ± 0.20                 | 4.89 ± 0.17                | 5.25 ± 0.04       | 3.27 ± 0.09      |
| Graph Wavenet  | 6.04 ± 0.28           | 4.41 ± 0.11            | 2.93 ± 0.09       | 10.50 ± 0.15     | 2.89 ± 0.03                 | 3.40 ± 0.05                | 5.29 ± 0.04       | 3.16 ± 0.09      |
| MTGNN          | 6.26 ± 0.18           | 4.46 ± 0.07            | 2.71 ± 0.23       | 9.55 ± 0.65      | 3.03 ± 0.03                 | 3.53 ± 0.03                | 5.39 ± 0.05       | 3.34 ± 0.02      |
| StemGNN        | 6.86 ± 0.28           | 4.76 ± 0.19            | 1.73 ± 0.02       | 7.71 ± 0.11      | 8.81 ± 0.37                 | 6.90 ± 0.02                | 5.75 ± 0.09       | 3.40 ± 0.09      |
| CrossGNN       | 7.22 ± 0.36           | 4.96 ± 0.12            | 2.95 ± 0.16       | 10.82 ± 0.21     | 3.03 ± 0.10                 | 3.48 ± 0.08                | 5.66 ± 0.04       | 3.53 ± 0.05      |
| FourierGNN     | 6.84 ± 0.35           | 4.65 ± 0.12            | 2.55 ± 0.03       | 10.22 ± 0.08     | 2.99 ± 0.02                 | 3.42 ± 0.02                | 5.82 ± 0.06       | 3.62 ± 0.07      |
| GRU-D          | 5.59 ± 0.09           | 4.08 ± 0.05            | 1.76 ± 0.03       | 7.53 ± 0.09      | 2.94 ± 0.05                 | 3.53 ± 0.06                | 5.54 ± 0.38       | 3.40 ± 0.28      |
| SeFT           | 9.22 ± 0.18           | 5.40 ± 0.08            | 1.87 ± 0.01       | 7.84 ± 0.08      | 12.20 ± 0.17                | 8.43 ± 0.07                | 5.80 ± 0.19       | 3.70 ± 0.11      |
| RainDrop       | 9.82 ± 0.08           | 5.57 ± 0.06            | 1.99 ± 0.03       | 8.27 ± 0.07      | 14.92 ± 0.14                | 9.45 ± 0.05                | 5.78 ± 0.22       | 3.67 ± 0.17      |
| Warpformer     | 5.94 ± 0.35           | 4.21 ± 0.12            | 1.73 ± 0.04       | 7.58 ± 0.13      | 2.79 ± 0.04                 | 3.39 ± 0.03                | 5.25 ± 0.05       | 3.23 ± 0.05      |
| mTAND          | 6.23 ± 0.24           | 4.51 ± 0.17            | 1.85 ± 0.06       | 7.73 ± 0.13      | 3.22 ± 0.07                 | 3.81 ± 0.07                | 5.33 ± 0.05       | 3.26 ± 0.10      |
| Latent-ODE     | 6.05 ± 0.57           | 4.23 ± 0.26            | 1.89 ± 0.19       | 8.11 ± 0.52      | 3.34 ± 0.11                 | 3.94 ± 0.12                | 5.62 ± 0.03       | 3.60 ± 0.12      |
| CRU            | 8.56 ± 0.26           | 5.16 ± 0.09            | 1.97 ± 0.02       | 7.93 ± 0.19      | 6.97 ± 0.78                 | 6.30 ± 0.47                | 6.09 ± 0.17       | 3.54 ± 0.18      |
| Neural Flow    | 7.20 ± 0.07           | 4.67 ± 0.04            | 1.87 ± 0.05       | 8.03 ± 0.19      | 4.05 ± 0.13                 | 4.46 ± 0.09                | 5.35 ± 0.05       | 3.25 ± 0.05      |
| tPatchGNN      | 4.98 ± 0.08           | 3.72 ± 0.03            | 1.69 ± 0.03       | 7.22 ± 0.09      | 2.66 ± 0.03                 | 3.15 ± 0.02                | **5.00 ± 0.04**   | *3.08 ± 0.04*    |
| GraFITi        | **4.89 ± 0.12**       | **3.65 ± 0.09**        | **1.53 ± 0.02**   | **6.68 ± 0.09**  | **2.65 ± 0.02**             | **3.09 ± 0.02**            | 5.17 ± 0.11       | 3.22 ± 0.24      |



To reproduce these results you can run:

```
python zhang_experiments.py --nlayers 4 --attn-head 2 --latent-dim 64 --dataset activity --history 3000
python zhang_experiments.py --nlayers 4 --attn-head 2 --latent-dim 64 --dataset mimic --history 24
python zhang_experiments.py --nlayers 4 --attn-head 2 --latent-dim 64 --dataset physionet --history 24
python zhang_experiments.py --nlayers 2 --attn-head 2 --latent-dim 128 --dataset ushcn --history 24
```
