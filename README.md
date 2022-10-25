# Seeing-Far-in-the-Dark-with-Patterned-Flash
Code and data release for the ECCV 2022 manuscript "[Seeing Far in the Dark with Patterned Flash]" (ECCV 2022)

[Zhanghao Sun], [Jian Wang], [Yicheng Wu], [Shree Nayar].

## [Paper], [Poster], [Video], [Supplementary]

[Seeing Far in the Dark with Patterned Flash]: https://arxiv.org/pdf/2207.12570.pdf
[Zhanghao Sun]: https://zhsun0357.github.io/
[Jian Wang]: https://jianwang-cmu.github.io/
[Yicheng Wu]: https://yichengwu.github.io/
[Shree Nayar]: http://www.cs.columbia.edu/~nayar/
[Paper]: https://arxiv.org/pdf/2207.12570.pdf
[Poster]: https://zhsun0357.github.io/data/2891.pdf
[Video]: https://zhsun0357.github.io/data/2891.mp4
[Supplementary]: https://jianwang-cmu.github.io/22patternedFlash/patteredFlash-supp.pdf

## Introduction
This repository is code release for our ECCV 2022 paper "Seeing Far in the Dark with Patterned Flash". 

Flash illumination is widely used in imaging under low-light environments. However, illumination intensity falls off with propagation distance quadratically, which poses significant challenges for flash imaging at a long distance. We propose a new flash technique, named "patterned flash", for flash imaging at a long distance. Patterned flash concentrates optical power into a dot array. Compared with the conventional uniform flash where the signal is overwhelmed by the noise everywhere, patterned flash provides stronger signals at sparsely distributed points across the field of view to ensure the signals at those points stand out from the sensor noise. This enables post-processing to resolve important objects and details. Additionally, the patterned flash projects texture onto the scene, which can be treated as a structured light system for depth perception. Given the novel system, we develop a joint image reconstruction and depth estimation algorithm with a convolutional neural network. We build a hardware prototype and test the proposed flash technique on various scenes. The experimental results demonstrate that our patterned flash has significantly better performance at long distances in low-light environments. 

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
Our code is adapted from the [KAIR] low-level computer vision library.

[KAIR]: https://github.com/cszn/KAIR

### Synthetic Dataset
We use RGB image (left) and depth map in [FlyingThings3D] dataset for both evaluation and training. 
We follow the image formation model in data synthesize process, including depth-dependent pattern warping and physics-based noise. We also notice that most current flash/no-flash reconstruction algorithms do not handle shadow correctly, since sharp shadow edges in no-flash images usually do not exist in flash images. We consider this effect with the stereo image pair provided by [FlyingThings3D] dataset.

After downloading the [FlyingThings3D] dataset, please organize it as follows (split files are avaiable [here]):

[here]: https://zhsun0357.github.io/

```
ROOT
|
--- train_split.txt                     # training set split file
--- val_split.txt                       # validation set split file
--- val_small_split.txt                 # small validation set split file
--- frames_cleanpass                    # color image folder
|       |
|       ---- TRAIN                      # training set
|       |      |
|       |      ---- A                   # subset
|       |      |      |
|       |      |      ---- 0000         # scene
|       |      |      |        +--- left         # left color images
|       |      |      |        +--- right        # right color images
|       |      |      .
|       |      |      .
|       |      |
|       |
|       +--- TEST
|       |      |
|       |      ---- A                   # subset
|
+-- disparity                           # disparity folder
|       .
|       .
```

One can run the evaluation for synthetic data with 

     python eval_pattern_flash.py --opt options/eval_pf.json 

#### Example Reconstruction Result
<img src='fig/results_synth.png'>

### Real-world Captured Data
We captured several images with our hardware prototype.

One can run the evaluation for real-world captured data with

     python eval_pattern_flash_real.py --opt options/eval_pf_real.json 

We also provide our pre-trained [checkpoint] of patterned flash reconstruction model.

[checkpoint]: https://zhsun0357.github.io/
[FlyingThings3D]: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

## Training
### Model training
One can run training with:
    
    python main_train_cnn_psnr_pattern_flash.py --opt options/train_pf.json
    
for patterned flash reconstruction model training or 

    python main_train_cnn_psnr_pattern_flash.py --opt options/train_fnf.json
        
for patterned flash/no-flash reconstruction model training
    
The whole training process has around 200k iters and would take around 36hrs on Nvidia Tesla V100 GPU (16GB).
You can easily change hyper-parameters and input/output file directories in the json files.

