# AesUST
**[update 8/28/2022]**

Official Pytorch code for ["AesUST: Towards Aesthetic-Enhanced Universal Style Transfer"](https://dl.acm.org/doi/pdf/10.1145/3503161.3547939) (ACM MM 2022)

## Introduction:

**AesUST** is a novel Aesthetic-enhanced Universal Style Transfer approach that can generate aesthetically more realistic and pleasing results for arbitrary styles. It introduces an aesthetic discriminator to learn the universal human-delightful aesthetic features from a large corpus of artist-created paintings. Then, the aesthetic features are incorporated to enhance the style transfer process via a novel Aesthetic-aware Style-Attention (AesSA) module. Moreover, we also develop a new two-stage transfer training strategy with two aesthetic regularizations to train our model more effectively, further improving stylization performance. 

![show](https://github.com/EndyWon/AesUST/blob/main/figures/teaser.jpg)

## Environment:
- Python 3.6
- Pytorch 1.8.0

## Getting Started:
**Clone this repo:**

`git clone https://github.com/EndyWon/AesUST`  
`cd AesUST`

**Test:**

- Download pre-trained models from this [google drive](https://drive.google.com/file/d/1Ldpfkt32r--ZWwhHSaKbJb7YJbOuwYLZ/view?usp=sharing). Unzip and place them at path `models/`.
- Test a pair of images:

  `python test.py --content inputs/content/1.jpg --style inputs/style/1.jpg`
  
- Test two collections of images:

  `python test.py --content_dir inputs/content/ --style_dir inputs/style/`

**Train:**

- Download content dataset [MS-COCO](https://cocodataset.org/#download) and style dataset [WikiArt](https://www.kaggle.com/c/painter-by-numbers) and then extract them.

- Download the pre-trained [vgg_normalised.pth](https://drive.google.com/file/d/1PUXro9eqHpPs_JwmVe47xY692N3-G9MD/view?usp=sharing), place it at path `models/`.

- Run train script:

  `python train.py --content_dir ./coco2014/train2014 --style_dir ./wikiart/train`


## Runtime Controls:

**Content-style trade-off:**

  `python test.py --content inputs/content/1.jpg --style inputs/style/1.jpg --alpha 0.5`
  
  ![show](https://github.com/EndyWon/AesUST/blob/main/figures/content_style_tradeoff.jpg)
  
**Style interpolation:**

  `python test.py --content inputs/content/1.jpg --style inputs/style/30.jpg,inputs/style/36.jpg --style_interpolation_weights 0.5,0.5`
  
  ![show](https://github.com/EndyWon/AesUST/blob/main/figures/style_interpolation.jpg)
  
**Color-preserved style transfer:**

  `python test.py --content inputs/content/1.jpg --style inputs/style/1.jpg --preserve_color`
  
  ![show](https://github.com/EndyWon/AesUST/blob/main/figures/color_preserved.jpg)
  


## Citation:

If you find the ideas and codes useful for your research, please cite the paper:

```
@inproceedings{wang2022aesust,
  title={AesUST: Towards Aesthetic-Enhanced Universal Style Transfer},
  author={Wang, Zhizhong and Zhang, Zhanjie and Zhao, Lei and Zuo, Zhiwen and Li, Ailin and Xing, Wei and Lu, Dongming},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)},
  year={2022}
}
```

## Acknowledgement:

We refer to some codes from [SANet](https://github.com/GlebSBrykin/SANET) and [IEContraAST](https://github.com/HalbertCH/IEContraAST). Great thanks to them!

