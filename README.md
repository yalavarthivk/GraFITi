# GraFITi

This is the source code for the paper ``GraFITi: Graphs for Forecasting of Irregularly sampled Time Series``


# Requirements
python                    3.8.11

Pytorch                   1.9.0

sklearn                   0.0

numpy                     1.19.3

pandas                    1.5

# Training and Evaluation

We provide an example for ``physionet`` for observing 36 hrs and predicting 12 hrs. All the datasets can be run in the similar manner.

```
train_grafiti.py --epochs 200 --learn-rate 0.001 --batch-size 128 --attn-head 1 --latent-dim 128 --nlayers 4 --dataset physionet2012 --fold 0 -ct 36 -ft 12
```

Remaining datasets can be run similarly. MIMIC-IV and MIMIC-III require permissions to download the data. Once, the datasets are downloaded, you can add them to the folder .tsdm/rawdata/ and use the TSDM package to extract the folds. We use TSDM package provided by Scholz .et .al from [https://openreview.net/forum?id=a-bD9-0ycs0]
