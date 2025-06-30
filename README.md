#   Lesion  Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing  and CNN-Transformer Fusion Networks  



## Introduction

This is an official release of the paper **Lesion  Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing  and CNN-Transformer Fusion Networks**.

<div align="center" border=> <img src=framework.jpg width="600" > </div>



## Code List

- [x] Network
- [x] Pre-processing
- [x] Training Codes




## Usage

### Dataset

Please download the dataset from [ISIC](https://www.isic-archive.com/) challenge website.

### Pre-processing

Please run:

```bash
$ python src/BinaryMapResize.py
$ python src/SkinAugment.py
$ python src/BinaryGroundPatchMap.py
```

You need to change the **File Path** to your own.



### Training 

### Testing

```bash
$ python test.py --dataset isic2018
```



## Citation

If you find MPBA-Net useful in your research, please consider citing:

```
@inproceedings{
  title={Lesion  Boundary Detection for Skin Lesion Segmentation Based on Boundary Sensing  and CNN-Transformer Fusion Networks},
  author={Xuzhen Huang, Yuliang Ma*, Xiajin Mei, ZizhuoWu, Mingxu Sun, Qingshan She}
}
```

