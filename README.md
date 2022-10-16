# AugTarget Data Augmentation


![](https://img.shields.io/badge/language-PyTorch-blue.svg?style=flat-square)


## [Datasets](#attention-guided-pyramid-context-networks)

- MDFA dataset is available at [MDvsFa cGan](https://github.com/wanghuanphd/MDvsFA_cGAN).
- SIRST dataset is available at [SIRST](https://github.com/YimianDai/sirst).
- The SIRST Augment dataset: download from [Google Drive](https://drive.google.com/file/d/13hhEwYHU19oxanXYf-wUpZ7JtiwY8LuT/view?usp=sharing) or [BaiduYun Drive](https://pan.baidu.com/s/1c35pADjPhkAcLwmU-u0RBA) with code `ojd4`.

## [Usage](#attention-guided-pyramid-context-networks)

### Train
```python
python train.py --net-name agpcnet_1 --batch-size 8 --save-iter-step 20 --dataset mdfa --learning-rate 0.05
```

```python
python train.py --net-name agpcnet_1 --batch-size 8 --save-iter-step 40 --dataset sirstaug --learning-rate 0.1
```

```python
python train.py --net-name agpcnet_1 --batch-size 8 --save-iter-step 100 --dataset merged --learning-rate 0.05
```

### Inference

```python
python inference.py --pkl-path {checkpoint path} --image-path {image path}
```

### Evaluation
```python
python evaluation.py --dataset {dataset name} 
                     --sirstaug-dir {base dir of sirstaug}
                     --mdfa-dir {base dir of MDFA}
                     --pkl-path {checkpoint path}
```


## [Results](#attention-guided-pyramid-context-networks)

| Methods | Data      | Precision | Recall | mIoU   | Fmeasure | AUC    | Download | 
| :---:   | :---:     | :---:     | :---:  | :---:  | :---:    | :---:  | :---:    |
| AGPCNet | MDFA      | 0.5939    | 0.7241 | 0.4843 | 0.6525   | 0.8682 | [model](./AGPCNet/mdfa_mIoU-0.4843_fmeasure-0.6525.pkl) |
| AGPCNet | SIRST Aug | 0.8323    | 0.8542 | 0.7288 | 0.8431   | 0.9344 | [model](./AGPCNet/sirstaug_mIoU-0.7288_fmeasure-0.8431.pkl) |
| AGPCNet | Merged    | 0.7453    | 0.8384 | 0.6517 | 0.7891   | 0.9194 | [model](./AGPCNet/merged_mIoU-0.6517_fmeasure-0.7891.pkl) |
| AGPCNet+AugTarget | MDFA      | 0.6482    | 0.7141 | 0.5146 | 0.6795   | 0.8699 | [model](./result/mdfa/mdfa_AugTarget.pkl) |
| AGPCNet+AugTarget  | SIRST Aug | 0.8449    | 0.8704 | 0.7505 | 0.8574   | 0.9378 | [model](./result/sirstaug/sirstaug_AugTarget.pkl) |
| AGPCNet+AugTarget  | Merged    | 0.7576    | 0.8658 | 0.6780 | 0.8081   | 0.9395 | [model](./result/merged/merged_AugTarget.pkl) |

## Acknowledgement
During implementation, our Target Augmentation algorithm is based on the random strategy of [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing), thanks for their contributions.
This repository is based on framework from [AGPCNet](https://github.com/Tianfang-Zhang/AGPCNet) and modified part of the code.


## Contact
If any questions, kindly contact with Shengjia Chen via e-mail: csj_uestc@126.com.







