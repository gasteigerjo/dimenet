# Directional Message Passing Neural Network (DimeNet and DimeNet++)

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/2dfilters_large_layer2.png?raw=true">
</p>


Reference implementation of the DimeNet model proposed in the paper:

**[Directional Message Passing for Molecular Graphs](https://www.daml.in.tum.de/dimenet)**   
by Johannes Klicpera, Janek Groß, Stephan Günnemann   
Published at ICLR 2020.

As well as its successor, DimeNet++.

## Run the code
This repository contains a notebook for training the model (`train.ipynb`) and for generating predictions on the test set with a trained model (`predict.ipynb`). It also contains a script for training the model on a cluster with Sacred and [SEML](https://github.com/TUM-DAML/seml) (`train_seml.py`). For faster experimentation we also offer two sets of pretrained models, which you can find in the `pretrained` folder.

## DimeNet++ and TF2

The new DimeNet++ model is both 10x faster and 10% more accurate, so we recommend using this model instead of the original. These improvements have not yet been published in a paper, but feel free to use it anyway (just cite the original paper). DimeNet++ was developed with the help of Shankari Giri.

There are some slight differences between this repository and the original (TF1) DimeNet model, such as slightly different training and initialization in TF2. This implementation uses orthogonal Glorot initialization in the output layer for the targets alpha, R2, U0, U, H, G, and Cv and zero initialization for Mu, HOMO, LUMO, and ZPVE. The paper only used zero initialization for the output layer.

The following table gives an overview of all MAEs:

<p align="left">
<img src="https://github.com/klicperajo/dimenet/blob/master/results_qm9_tf2_pp.svg?raw=true&sanitize=true">
</p>

## Architecture

### DimeNet

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/architecture.svg?raw=true&sanitize=true">
</p>

### DimeNet++

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/architecture_pp.svg?raw=true&sanitize=true">
</p>

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
