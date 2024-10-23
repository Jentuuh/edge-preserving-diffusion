# Edge-preserving noise for diffusion models
This repository contains the code for our paper:

> **Edge-preserving noise for diffusion models** | [[Project Page]](https://edge-preserving-diffusion.mpi-inf.mpg.de/)
>
> Jente Vandersanden, Sascha Holl, Xingchang Huang, Gurprit Singh
> 
> ArXiv, 2024

![teaser](assets/teaser.png)

## 📂 Code structure

* `assets/`: README assets
* `configs/`: Configuration files. We provided the configuration files for several datasets that are preset to the configurations we used for our final experiments.
* `utils/`: Helper files. `tools.py` contains some general tools for things like I/O and type conversions. `unet.py` and `nn.py` are borrowed from the [Improved DDPM codebase](https://github.com/openai/improved-diffusion?tab=readme-ov-file) and specify model architecture. Finally `vanilla_ddpm_model.py` and `edge_preserving_model.py` provide an implementation of vanilla DDPM and our edge-preserving model, respectively.
* Finally, the files `train.py` and `sample.py` provide a minimalistic implementation of a training and sampling loop. We provide examples of how to use these scripts below.
 
## 🛠️ Installation
The following are tested on Linux, with an NVIDIA GeForce RTX 3090 and CUDA 11.8 installed.

First, make sure you install Anaconda [here](https://docs.anaconda.com/anaconda/install/) if you haven't. Then simply run the following command, which should create and initialize an environment with the required dependencies installed:
```
conda create --name <YOUR_ENVIRONMENT_NAME> --file requirements.txt
```

## 🗝️ Usage

We provide minimalistic code for both vanilla DDPM as well as the proposed edge-preserving model, for convenient comparison. To train/sample with DDPM, specify flag `--method "ddpm"`. To use the edge-preserving model, specify `--method "ours"`. 

### 🚀 Training example
```
python3 train.py --config configs/afhq_cat_128/afhq_cat_config.py --workdir ./experiments/cat_training_example --datadir <PATH_TO_YOUR_DATASET> --method "ddpm" 
```

### 🧪 Sampling example
```
python3 sample.py --config configs/afhq_cat_128/afhq_cat_config.py --checkpointdir <PATH_TO_YOUR_CHECKPOINT_FILE> --savedir ./samples/cat_sampling_example --method "ours" --n_samples 5 --batch_size 1
```

## 👍 Citation
If you find this code useful please consider citing:
```
@inproceedings{vandersanden2024edge,
  title={Edge-preserving noise for diffusion models},
  author={Vandersanden, Jente and Holl, Sascha and Huang, Xingchang and Singh, Gurprit},
  booktitle={ArXiv},
  year={2024}
}
```

## 🔒 License
Edge-preserving noise for diffusion models is distributed under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license. The license applies to the pre-trained models and the metadata as well.
