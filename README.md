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
We use Python 3.6.7 , Pytorch 1.9 and CUDA 10.2 for our experiments. One can install our conda environment from "environment.yml".

## Evaluation
### Simulated Dataset
One can evaluate SPADnet model on simulated NYUV2 dataset by running:
    
    python evaluate_spadnet.py

This will create a .json file that contains all metrices of evaluated model.
You can change hyper-parameters and input/output file directories in "val_config.ini"

## Training
### Model training
We provide our training scipts for both patterned flash reconstruction and patterned flash/no-flash reconstruction. One can run training with:
    
    python main_train_cnn_psnr_pattern_flash.py --opt options/train_pf.json
    
for patterned flash reconstruction model training or 

    python main_train_cnn_psnr_pattern_flash.py --opt options/train_fnf.json
        
for patterned flash/no-flash reconstruction model training
    
after both SPAD simulation and corresponding monocular depth estimations are completed. We use Adam Optimizer, with a learning rate of 1e-4 and learning rate decay of 0.5 after each epoch. The whole training process has 5 epochs and would take around 24hrs on Nvidia Titan V GPU (12GB).
You can easily change hyper-parameters and input/output file directories in "config.ini"

We also provide a pre-trained snapshot of SPADnet model in "pth" folder (12.5MB).

[NYUV2]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html


