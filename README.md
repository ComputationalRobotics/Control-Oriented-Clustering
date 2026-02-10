# Control-oriented Clustering of Visual Latent Representation (ICLR2025 Spotlight)

This repository contains the code for the paper
"Control-Oriented Clustering of Visual Latent Representations".
It provides training, evaluation, and analysis pipelines for studying
Neural Collapse (NC) in vision-based control policies, as well as
control-oriented pretraining strategies.

======================================================================

## Environment Setup

Create the conda environment using the provided configuration file:

    conda env create -f environment.yml

The code is developed and tested with:
- Python: 3.12.5
- CUDA: 12.4

Make sure your system configuration is compatible.

======================================================================

## Prevalent Neural Collapse in Latent Representation Space

1. Train a Vision-Based Control Policy

We provide an example implementation using:
- ResNet-18 as the vision encoder
- Diffusion model as the action decoder

The corresponding dataset used in the paper is also included.

Run the following command in the training folder:

```bash
    python train_model.py
```

Notes:
- The default setting trains the model for 300 epochs, which matches
  the configuration used in the paper.
- Model checkpoints are automatically saved and later used for
  Neural Collapse evaluation.


2. Test the Trained Model

Run the following command in the evaluation_test_score folder:

```bash
    python test_domain_18_model.py
```

Notes:
- By default, the script evaluates checkpoints from 20 different epochs
  across the full 300-epoch training process.


3. Evaluate Neural Collapse (NC)

Two classification (labeling) strategies are provided, as described
in the paper.

a. Goal-Based Classification (Input Space)

Run the following command in the
observe_NC_metric_input_space_labeling folder:

```bash
    python domain_18_observe_NC_metric_input_space_labeling.py
```

This script computes NC metrics for saved checkpoints across
different training epochs.

b. Action-Based Classification (Action Space)

Run the following command in the
observe_NC_metric_action_intention_labeling folder:

```bash
    python domain_18_observe_NC_metric_action_intention_labeling.py
```

This script evaluates NC metrics for saved checkpoints at
different epochs.

======================================================================

## Control-Oriented Pretraining

Control-oriented pretraining code is provided in the NC_pretraining
folder.

Step 1: Pretrain the vision encoder

```bash
    python NC_pretrain.py
```

Step 2: End-to-end training of the vision encoder and diffusion model

```bash
    python NC_together.py
```

This two-stage training strategy follows the procedure described
in the paper.

======================================================================

## Letters Planar Pushing Dataset

We release the Letters Planar Pushing dataset used in this paper. The dataset is publicly available at:
https://github.com/han20192019/Letters-Planar-Pushing-Dataset



======================================================================
## Citation

If you find this dataset useful, please cite:

```bibtex
@inproceedings{qi25iclr-control,
title={Control-oriented Clustering of Visual Latent Representation},
author={Qi, Han and Yin, Haocheng and Yang, Heng},
booktitle={International Conference on Learning Representations (ICLR)},
year={2025},
note={\url{https://arxiv.org/abs/2410.05063}, \url{https://computationalrobotics.seas.harvard.edu/ControlOriented_NC/}}
}
```