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
[Download the file](https://drive.google.com/drive/folders/17bJQ609VfV_i0OiqB0-xkw6TwfkGE4Mp?usp=sharing) and place it in the `weights` folder.  


## 🔧Usage
### Training
```python
python train.py --name train --size_rounded --batchSize 1 --base_dir <YOUR_DATA_DIR>
```
### Testing
```python
python eval.py --name eval --size_rounded --test_nature --weight_path <YOUR_WEIGHT_PATH> --base_dir <YOUR_DATA_DIR>
```

## 🧪Figure 1 Reproduction and Image Generation Notes

The OpenRR-1k Figure 1 teaser image discussed in the paper corresponds to
`val_100/val_0034.jpg`. This image should be treated as a validation/test image, not as a
training sample.

In the reproduction workflow, the model is fine-tuned on **OpenRR-1k `train_800`** and then
evaluated on **OpenRR-1k `val_100/val_0034.jpg`** to generate the Figure 1-style visual panels.
The released checkpoint alone, when tested zero-shot on `val_0034`, does not reproduce the
28.54 dB teaser result; that value is reached only after short in-domain fine-tuning on the
OpenRR-1k training split.

### Image generation workflow

The Figure 1-style panels are generated in three steps:

1. Fine-tune ReflexSplit on `train_800`.
2. Save the transmission and reflection outputs for `val_0034` at selected iterations.
3. Assemble the saved outputs into comparison panels.

Expected per-iteration outputs:

```text
iterXXX_T.png   # predicted transmission / reflection-removed image
iterXXX_R.png   # predicted reflection layer
```

Expected assembled outputs:

```text
fig1_iterXXX.png   # paper-style panel: input/GT, DSIT, RDNet, ReflexSplit
grid.png           # iteration overview, e.g. iter 140-160
```

Example commands used by the reproduction package:

```bash
PYTHONPATH=/path/to/ReflexSplit OUT_DIR=/data/out/rs \
REFLEX_NO_HIST=1 REFLEX_PAPER_LOSS=1 REFLEX_AMP=1 \
python code/ft_reflexsplit.py

python code/make_fig1.py \
  --rs_dir /data/out/rs \
  --dsit_dir /data/out/dsit \
  --rdnet_dir /data/out/rdnet \
  --rs_log /data/out/rs.log \
  --dsit_log /data/out/dsit.log \
  --rdnet_log /data/out/rdnet.log \
  --inp /data/openrr1k/val_100/val_100/blended/val_0034.jpg \
  --gt /data/openrr1k/val_100/val_100/transmission_layer/val_0034.jpg \
  --out /data/out/fig1
```

### Important metric note

When comparing ReflexSplit with DSIT and RDNet, all methods must use the same image conversion
and PSNR calculation. The model outputs are in `[0, 1]`, so they should be converted by clipping
to `[0, 1]` and multiplying by 255. Do not mix visualization utilities from different
repositories, because inconsistent normalization can produce washed-out images and inflated PSNR.

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

## 
This study was supported in part by the National Science and Technology Council (NSTC), Taiwan, under grants 112-2221-E-006-157-MY3, 114-2627-M-A49-003, 114-2218-E-035-001, and 114-2119-M-006-007. We thank to National Center for High-performance Computing (NCHC) of National Applied Research Laboratories (NARLabs) in Taiwan for providing computational and storage resources.
