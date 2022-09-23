# Seeing-Far-in-the-Dark-with-Patterned-Flash
Code and data release for the ECCV 2022 manuscript "[Seeing Far in the Dark with Patterned Flash]" (ECCV 2022)

[Zhanghao Sun], [Jian Wang], [Yicheng Wu], [Shree Nayar].

## [Poster], [Video], [Supplementary Material]

[Seeing Far in the Dark with Patterned Flash]: https://arxiv.org/pdf/2207.12570.pdf
[Zhanghao Sun]: https://zhsun0357.github.io/
[Jian Wang]: https://jianwang-cmu.github.io/
[Yicheng Wu]: https://yichengwu.github.io/
[Shree Nayar]: http://www.cs.columbia.edu/~nayar/
[Poster]: https://zhsun0357.github.io/data/2891.pdf
[Video]: https://zhsun0357.github.io/data/2891.mp4
[Supplementary Material]: https://jianwang-cmu.github.io/22patternedFlash/patteredFlash-supp.pdf

## Introduction
This repository is code release for our ECCV 2022 paper "Seeing Far in the Dark with Patterned Flash". 



For more details of our work, please refer to our technical paper.

## Citation
If you find our work useful in your research, please consider citing:

        @article{sun2022seeing,
          title={Seeing Far in the Dark with Patterned Flash},
          author={Sun, Zhanghao and Wang, Jian and Wu, Yicheng and Nayar, Shree},
          journal={arXiv preprint arXiv:2207.12570},
          year={2022}
        }

## Installation
We use Python 3.6 , Pytorch 1.0 and CUDA 9.0 for our experiments. One can install our conda environment from "environment.yml".

## Training
### Preparing data
The data preparation process contains SPAD simualtion and corresponding monocular depth estimation. We use [NYUV2] dataset for SPAD measurement simulation. We select out data with high quality (without large holes in ground truth depth map, with reasonable reflectivity value and so on), which are separated into training set, validation set and test set (10:1:1). Corresponding scene index are listed in "util/train_clean.txt", "util/val_clean.txt" and "util/test_clean.txt".

To simulate SPAD measurements, we adapted code from NYUV2 toolkit and code from [Lindell et al., 2018]. The signal-background ratio (SBR) needs to be specified for simulation. We always use the lowest SBR (level 9, which corresponds to 2 signal photons and 50 background photons) during experiments and observed good generalization capability to complicated real-world scenes.

Our scripts directly load monocular estimation results. We use [DORN] model as monocular estimation network for most part of the work and [here] we provide corresponding estimation results. Users can replace them with any other preliminary depth estimations.

[Lindell et al., 2018]: http://www.computationalimaging.org/publications/single-photon-3d-imaging-with-deep-sensor-fusion/
[DORN]: https://openaccess.thecvf.com/content_cvpr_2018/html/Fu_Deep_Ordinal_Regression_CVPR_2018_paper.html
[here]: https://drive.google.com/file/d/1bHpdTCIARwOazWa7Up3o31hrGDmwetj4/view?usp=sharing

### Model training
One can train SPADnet model from scratch by running:
    
    python train_spadnet.py
    
after both SPAD simulation and corresponding monocular depth estimations are completed. We use Adam Optimizer, with a learning rate of 1e-4 and learning rate decay of 0.5 after each epoch. The whole training process has 5 epochs and would take around 24hrs on Nvidia Titan V GPU (12GB).
You can easily change hyper-parameters and input/output file directories in "config.ini"

We also provide a pre-trained snapshot of SPADnet model in "pth" folder (12.5MB).

[NYUV2]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

## Evaluation
### Simulated Dataset
One can evaluate SPADnet model on simulated NYUV2 dataset by running:
    
    python evaluate_spadnet.py

This will create a .json file that contains all metrices of evaluated model.
You can change hyper-parameters and input/output file directories in "val_config.ini"

