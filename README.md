# edge-preserving-diffusion
Source code repository for the project "Edge-preserving noise for diffusion models", ArXiv 2024.


This repository contains code for both vanilla DDPM as well as the proposed edge-preserving model, for convenient comparison. To train/sample with DDPM, specify flag `--method "ddpm"`. To use the edge-preserving model, specify `--method "ours"`. 

## Training example
```
    python3 train.py --config configs/afhq_cat_128/afhq_cat_config.py --workdir ./experiments/cat_training_example --datadir <PATH_TO_YOUR_DATASET> --method "ddpm" 
```

## Sampling example
```
    python3 sample.py --config configs/afhq_cat_128/afhq_cat_config.py --checkpointdir <PATH_TO_YOUR_CHECKPOINT_FILE> --savedir ./samples/cat_sampling_example --method "ours" --n_samples 5 --batch_size 1
```