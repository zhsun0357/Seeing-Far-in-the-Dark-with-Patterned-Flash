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
### Synthetic Dataset
We use RGB image (left) and depth map in [FlyingThings3D] dataset for both evaluation and training. 
We follow the image formation model in data synthesize process, including depth-dependent pattern warping and physics-based noise. We also notice that most current flash/no-flash reconstruction algorithms do not handle shadow correctly, since sharp shadow edges in no-flash images usually do not exist in flash images. We consider this effect with the stereo image pair provided by [FlyingThings3D] dataset.

One can run the evaluation for synthetic data with 

     python eval_pattern_flash.py --opt options/eval_pf.json 

### Real-world Captured Data
We captured several images with our hardware prototype.

One can run the evaluation for real-world captured data with

     python eval_pattern_flash_real.py --opt options/eval_pf_real.json 

[FlyingThings3D]: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

## Training
### Model training
One can run training with:
    
    python main_train_cnn_psnr_pattern_flash.py --opt options/train_pf.json
    
for patterned flash reconstruction model training or 

    python main_train_cnn_psnr_pattern_flash.py --opt options/train_fnf.json
        
for patterned flash/no-flash reconstruction model training
    
The whole training process has around 200k iters and would take around 36hrs on Nvidia Tesla V100 GPU (16GB).
You can easily change hyper-parameters and input/output file directories in the json files

We also provide a pre-trained [checkpoint of patterned flash reconstruction model].

[checkpoint of patterned flash reconstruction model]: https://zhsun0357.github.io/

