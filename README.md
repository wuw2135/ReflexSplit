<p align="center">
  <img src="assets/icon.jpg" alt="icon" />
</p>

# 🪞ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation (✨CVPR 2026✨)

<div align="center">

<!-- Badges row -->
<p>
  <a href="https://wuw2135.github.io/ReflexSplit-ProjectPage/">
    <img src="https://img.shields.io/badge/ProjectPage-⏎-80c1e0?logo=github&logoColor=80c1e0" />
  </a>
  <a href="https://www.arxiv.org/pdf/2601.17468">
    <img src="https://img.shields.io/badge/arXiv-2601.17468-red?logo=arxiv&logoColor=red" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/System-Ubuntu-f47421.svg?logo=Ubuntu&logoColor=f47421" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10-blue.svg?logo=python&logoColor=3776AB" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/pytorch->=2.0-%237732a8?logo=PyTorch&color=EE4C2C" />
  </a>
</p>

<!-- Authors -->
<p style="font-size: 15px;">
  <a href="https://ming053l.github.io/">Chia-Ming Lee</a>,
  <a href="https://vanlinlin.github.io/">Yu-Fan Lin</a>,
  <a href="https://github.com/wuw2135">Jin-Hui Jiang</a>,
  <a href="https://github.com/yujouhsiao">Yu-Jou Hsiao</a>,
  <a href="https://cchsu.info/">Chih-Chung Hsu</a>,
  <a href="https://yulunalexliu.github.io/">Yu-Lun Liu</a>
</p>

</div>



### Framework
<p align="center">
  <img src="assets/Reflex_arch.png" alt="framework" />
</p>

### Layer Fusion-Separation Block (LFSB) 
<p align="center">
  <img src="assets/Reflex_DDAIB.jpg" alt="lfsb" />
</p>

<details style="font-size: 15px;">
<summary>Abstract</summary> 
Single Image Reflection Separation (SIRS) disentangles mixed images into transmission and reflection layers. Existing methods suffer from transmission-reflection confusion under nonlinear mixing, particularly in deep decoder layers, due to implicit fusion mechanisms and inadequate multi-scale coordination. We propose ReflexSplit, a dual-stream framework with three key innovations. (1) Cross scale Gated Fusion (CrGF) adaptively aggregates semantic priors, texture details, and decoder context across hier archical depths, stabilizing gradient flow and maintaining feature consistency. (2) Layer Fusion-Separation Blocks (LFSB) alternate between fusion for shared structure extraction and differential separation for layer-specific disentanglement. Inspired by Differential Transformer, we extend attention cancellation to dual-stream separation via cross-stream subtraction. (3) Curriculum training progressively strengthens differential separation through depth dependent initialization and epoch-wise warmup. Extensive experiments on synthetic and real-world benchmarks demonstrate state-of-the-art performance with superior perceptual quality and robust generalization.
</details>

## 🏞️Environment
### Installation
```
pip install torch>=2.0 torchvision
pip install numpy scipy scikit-learn matplotlib opencv-python tqdm einops tensorboardx tensorboard dominate
```

## 🗂️Data preparing
### Data Structure
```
Datasets/
├── dataset1/
│   ├── blended/
|   |   ├── 1.png
|   |   ├── 2.png
|   |   ...
│   ├── reflection_layer/
|   |   ├── 1.png
|   |   ├── 2.png
|   |   ...
│   └── transmission_layer/
|       ├── 1.png
|       ├── 2.png
|       ...
├── dataset2/
│   ├── blended/
|   |   ├── 1.png
|   |   ├── 2.png
|   |   ...
│   └── transmission_layer/
|       ├── 1.png
|       ├── 2.png
|       ...
...
```
**⚠️ If you use the SIR² dataset, please follow the structure pattern of dataset1; for all others, please follow the structure pattern of dataset2.**

### Training dataset
* 7,643 images from the
  [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/), center-cropped as 224 x 224 slices to synthesize training pairs;  
* 90 real-world training pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal);
* 200 real-world training pairs provided by [IBCLN](https://github.com/JHL-HUST/IBCLN);

### Testing dataset
* 45 real-world testing images from [CEILNet dataset](https://github.com/fqnchina/CEILNet);
* 20 real testing pairs provided by [Zhang *et al.*](https://github.com/ceciliavision/perceptual-reflection-removal);
* 20 real testing pairs provided by [IBCLN](https://github.com/JHL-HUST/IBCLN);
* 500 real testing pairs from [SIR^2 dataset](https://sir2data.github.io/), containing three subsets (i.e., Objects (200), Postcard (199), Wild (101)). 

### Trained weights
to be continued...
## 🔧Usage
### Training
```python
python train.py --name train --size_rounded --batchSize 1 --base_dir <YOUR_DATA_DIR>
```
### Testing
```python
python eval.py --name eval --size_rounded --test_nature --weight_path <YOUR_WEIGHT_PATH> --base_dir <YOUR_DATA_DIR>
```

## 🎭Visual Comparison
<p align="center">
  <img src="assets/vis.png" alt="vis" />
</p>

## 📊Quantitative Comparison
<p align="center">
  <img src="assets/compare.png" alt="compare" />
</p>

## 🔬Citation
```
@article{lee2026reflexsplit,
  title={ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation},
  author={Lee, Chia-Ming and Lin, Yu-Fan and Jiang, Jin-Hui and Hsiao, Yu-Jou and Hsu, Chih-Chung and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2601.17468},
  year={2026}
}
```





