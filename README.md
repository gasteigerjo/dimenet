# Directional Message Passing Neural Network (DimeNet)

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/2dfilters_large_layer2.png?raw=true">
</p>


Reference implementation of the DimeNet model proposed in the paper:

**[Directional Message Passing for Molecular Graphs](https://www.daml.in.tum.de/dimenet)**   
by Johannes Klicpera, Janek Groß, Stephan Günnemann   
Published at ICLR 2020.

## TensorFlow 2

For this implementation we have migrated the original code to TensorFlow 2. The predictions are the same but we observe an increased runtime of 70% compared to the original TensorFlow 1 implementation, despite extensive efforts to mitigate this. Since GPU profiling is currently broken in TensorFlow 2.1 (see https://github.com/tensorflow/tensorboard/issues/3256), we can't investigate the issue further. We will have to wait until profiling is fixed or a new, faster TensorFlow version comes out.

## Run the code
This repository contains a notebook for training the model (`train.ipynb`) and for generating predictions on the test set with a trained model (`predict.ipynb`). It also contains a script for training the model on a cluster with Sacred and [SEML](https://github.com/TUM-DAML/seml) (`train_seml.py`).

## Architecture

<p align="center">
<img src="https://github.com/klicperajo/dimenet/blob/master/architecture.svg?raw=true&sanitize=true">
</p>

## Requirements
The repository uses these packages:

```
numpy
scipy
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
