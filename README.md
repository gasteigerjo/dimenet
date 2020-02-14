# [WIP] Directional Message Passing Neural Network (DimeNet)

<p align="center">
<img src="https://raw.githubusercontent.com/klicperajo/dimenet/master/2dfilters_large_layer2.png">
</p>


Reference implementation of the DimeNet model proposed in the paper:

**[Directional Message Passing for Molecular Graphs](https://www.kdd.in.tum.de/gdc)**   
by Johannes Klicpera, Janek Groß, Stephan Günnemann   
Published at ICLR 2020.

## WORK IN PROGRESS

This repository is still **work in progress**, since we are currently migrating the model to Tensorflow 2.0.

<!-- ## Run the code
This repository primarily contains a demonstration of enhancing a graph convolutional network (GCN) with graph diffusion convolution (GDC) in the notebook `gdc_demo.ipynb`. -->

## Requirements
The repository uses these packages:

```
numpy
scipy
sympy
tensorflow>=2.0
```

## Contact
Please contact klicpera@in.tum.de in case you have any questions.

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
