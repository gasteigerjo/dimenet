# Directional Message Passing Neural Network (DimeNet and DimeNet++)

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/2dfilters_large_layer2.png?raw=true">
</p>


Reference implementation of the DimeNet model proposed in the paper:

**[Directional Message Passing for Molecular Graphs](https://www.daml.in.tum.de/dimenet)**   
by Johannes Klicpera, Janek Groß, Stephan Günnemann   
Published at ICLR 2020.

As well as DimeNet++, its significantly faster successor:

**[Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://www.daml.in.tum.de/dimenet)**   
by Johannes Klicpera, Shankari Giri, Johannes T. Margraf, Stephan Günnemann   
Published at the ML for Molecules workshop, NeurIPS 2020.

## Run the code
This repository contains a notebook for training the model (`train.ipynb`) and for generating predictions on the test set with a trained model (`predict.ipynb`). It also contains a script for training the model on a cluster with Sacred and [SEML](https://github.com/TUM-DAML/seml) (`train_seml.py`). For faster experimentation we also offer two sets of pretrained models, which you can find in the `pretrained` folder.

## DimeNet++ and TF2

The new DimeNet++ model is both 8x faster and 10% more accurate, so we recommend using this model instead of the original.

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

## Known issues

Unfortunately there are a few issues/bugs in the code (and paper) that we can't fix without retraining the models. So far, these are:
- The second distance used for calculating the angles is switched ([DimeNet](https://github.com/klicperajo/dimenet/blob/master/dimenet/model/dimenet.py#L89)).
- The envelope function is implicitly divided by the distance. This is accounted for in the radial bessel basis layer but leads to an incorrect spherical basis  layer ([DimeNet and DimeNet++](https://github.com/klicperajo/dimenet/blob/master/dimenet/model/layers/envelope.py#L21)).
- DimeNet was evaluated on MD17's Benzene17 dataset, but compared to sGDML on Benzene18, which gives sGDML an unfair advantage.
- In TensorFlow AddOns <0.12 there is a bug when checkpointing. The earlier versions require explicitly passing the `_optimizer` variable of the `MovingAverage` optimizer. This is only relevant if you actually load checkpoints from disk and continue training ([DimeNet and DimeNet++](https://github.com/klicperajo/dimenet/blob/master/train_seml.py#L182)).
- The radial basis functions in the interaction block actually use d_kj and not d_ji. The best way to fix this is by just using d_ji instead of d_kj in the SBF and leaving the RBF unchanged ([DimeNet and DimeNet++](https://github.com/klicperajo/dimenet/blob/master/dimenet/model/layers/interaction_pp_block.py#L59)).

## Contact
Please contact klicpera@in.tum.de if you have any questions.

## Cite
Please cite our papers if you use the model or this code in your own work:

```
@inproceedings{klicpera_dimenet_2020,
  title = {Directional Message Passing for Molecular Graphs},
  author = {Klicpera, Johannes and Gro{\ss}, Janek and G{\"u}nnemann, Stephan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year = {2020}
}

@inproceedings{klicpera_dimenetpp_2020,
title = {Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules},
author = {Klicpera, Johannes and Giri, Shankari and Margraf, Johannes T. and G{\"u}nnemann, Stephan},
booktitle={NeurIPS-W},
year = {2020} }
```
