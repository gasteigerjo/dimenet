# Directional Message Passing Neural Network (DimeNet)

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/2dfilters_large_layer2.png?raw=true">
</p>


Reference implementation of the DimeNet model proposed in the paper:

**[Directional Message Passing for Molecular Graphs](https://www.daml.in.tum.de/dimenet)**   
by Johannes Klicpera, Janek Groß, Stephan Günnemann   
Published at ICLR 2020.

## Run the code
This repository contains a notebook for training the model (`train.ipynb`) and for generating predictions on the test set with a trained model (`predict.ipynb`). It also contains a script for training the model on a cluster with Sacred and [SEML](https://github.com/TUM-DAML/seml) (`train_seml.py`).

## Architecture

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/architecture.svg?raw=true&sanitize=true">
</p>

## Pretrained models

For faster experimentation we offer a set of pretrained models, which you can find in the `pretrained` folder. On average, these models _outperform_ the results reported in the paper by 3% (see table below).

This difference is due to slightly different training and initialization in TF2 and to using orthogonal Glorot initialization in the output layer for the targets alpha, R2, U0, U, H, G, and Cv, while using zero initialization for Mu, HOMO, LUMO, and ZPVE. The paper used the exact same architecture and hyperparameters in all experiments. It only used zero initialization for the output layer.

<p align="left">
<img src="https://github.com/klicperajo/dimenet/blob/master/results_qm9_tf2.svg?raw=true&sanitize=true">
</p>

## Training time

Training the original DimeNet architecture is rather slow (around 20 days for 3M steps on an Nvidia GTX 1080Ti). We are currently working on reducing this and have so far achieved a 10x speedup while further improving the accuracy. Feel free to contact us for details if you are interested in using DimeNet for a novel application.

## Requirements
The repository uses these packages:

```
numpy
scipy>=1.3
sympy>=1.5
tensorflow>=2.1
tensorflow_addons
tqdm
```

## Contact
Please contact klicpera@in.tum.de if you have any questions.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{klicpera_dimenet_2020,
  title = {Directional Message Passing for Molecular Graphs},
  author = {Klicpera, Johannes and Gro{\ss}, Janek and G{\"u}nnemann, Stephan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year = {2020}
}
```
