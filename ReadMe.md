## What does it do?
It does horizontal (rowwise) correlation of two tensors (`B x C x H x W`), which is useful for stereo
matching. A straight look at the math explains it well. **L** and **R** are
the left and right tensor. `horcorr` is the implemented function.
In the following we only consider one row, as generalization to batches and multiple rows is simple.

![Math][math]
## Why?
At the moment of writing this there is no possibility to implement this
in pure pytorch (version 1.2) without using up a huge amount of memory.
So there are other implementations which do something similiar (I think), e.g.
[Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension). Another reason was to learn about implementing
custom pytorch ops with gradient support.
## Features
 * horizontal (rowwise) correlation
 * efficient cuda implementation
 * calculates gradients for left and right image

## Installation
Edit the arguments `DIM`, `NWINDOWS` and `BLOCKSIZE` in `setup.py`.
Then execute `setup.py` as needed, e.g.

```
python setup.py bdist_wheel
```
## Example
A stereo [image](https://commons.wikimedia.org/wiki/Category:Stereo_images#/media/File:Cambridge_Kings_Hedges_Golden_Hind_RL.jpg) is preprocessed so that
subwindows `(N x N)` of the image `(H x W x C)` make up the channels of the block-image
`(H x W x (N * N * C))`. The left and the right half are than correlated horizontally. The visualization shows the disparity where the correlation is
maximized. What can be seen is that dominant regions maximize the correlation
over about one third of the image width. To calculate a disparity image an
algorithm would have to perform some kind of assignment to prevent disparity collisions.
(The module was built with `DIM=147`, `NWINDOWS=70` and `BLOCKSIZE=64` for the example.)

![Example][example]

[example]: example.jpg
[math]: math.png