#   FastTrans-Net: Fusion network with shift window hierarchical attention in multi-level enhanced space for skin lesion segmentation  



## Introduction

This is an official release of the paper **FastTrans-Net:  Fusion network with shift window hierarchical attention in multi-level  enhanced space for skin lesion segmentation**.

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
  title={FastTrans-Net: Fusion network with shift window hierarchical attention in multi-level enhanced space for skin lesion segmentation},
  author={Xuzhen Huang, Yuliang Ma*, Xiajin Mei, Mingxu Sun, Shouqiang Jia}
}
```

