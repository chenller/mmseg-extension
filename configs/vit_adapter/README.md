# ViT-Adapter

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=vision-transformer-adapter-for-dense)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-transformer-adapter-for-dense/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=vision-transformer-adapter-for-dense)

The official implementation of the
paper "[Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534)".

[Paper](https://arxiv.org/abs/2205.08534) | [Blog in Chinese](https://zhuanlan.zhihu.com/p/608272954) | [Slides](https://drive.google.com/file/d/1LotIZIEnZzKhsANjBTZs3qcezk9fbVCV/view?usp=share_link) | [Poster](https://iclr.cc/media/PosterPDFs/ICLR%202023/12048.png?t=1680764158.7068026) | [Video in English](https://iclr.cc/virtual/2023/poster/12048) | [Video in Chinese](https://www.bilibili.com/video/BV1ry4y1976b/?spm_id_from=333.337.search-card.all.click)

[Segmentation Colab Notebook](https://colab.research.google.com/drive/1yEd5lQMjShloicImtShkwttb74KPGY5U?usp=sharing) | [Detection Colab Notebook](https://colab.research.google.com/drive/1Im7l0dSvEgsP-AJtUOxgbU9a1C3DdSwe?usp=sharing) (
thanks [@IamShubhamGupto](https://github.com/IamShubhamGupto), [@dudifrid](https://github.com/dudifrid))

## Abstract

This work investigates a simple yet powerful dense prediction task adapter for Vision Transformer (ViT). Unlike recently
advanced variants that incorporate vision-specific inductive biases into their architectures, the plain ViT suffers
inferior performance on dense predictions due to weak prior assumptions. To address this
issue, we propose the ViT-Adapter, which allows plain ViT to achieve comparable performance to vision-specific
transformers. Specifically, the backbone in our
framework is a plain ViT that can learn powerful representations from large-scale
multi-modal data. When transferring to downstream tasks, a pre-training-free
adapter is used to introduce the image-related inductive biases into the model,
making it suitable for these tasks. We verify ViT-Adapter on multiple dense prediction tasks, including object
detection, instance segmentation, and semantic segmentation. Notably, without using extra detection data, our
ViT-Adapter-L yields
state-of-the-art 60.9 box AP and 53.0 mask AP on COCO test-dev. We hope that
the ViT-Adapter could serve as an alternative for vision-specific transformers and
facilitate future research. The code and models will be released.

## Method

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/217998186-8a37eacb-18f8-445a-8d92-0863e35712ab.png">

<img width="810" alt="image" src="https://user-images.githubusercontent.com/23737120/194904786-ea9c40a3-f6ac-4fe1-90ad-976e7b9e8f03.png">

## Segmentation Results
### [ViT-Adapter](https://github.com/czczup/ViT-Adapter)
You can find detailed information about ViT Adapters in [README.md](https://github.com/czczup/ViT-Adapter/blob/main/segmentation/README.md).

#### Pretraining Sources

| Name          | Year | Type       | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           | Support? |
|---------------|------|------------|--------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| DeiT          | 2021 | Supervised | ImageNet-1K  | [repo](https://github.com/facebookresearch/deit/blob/main/README_deit.md)                               | [paper](https://arxiv.org/abs/2012.12877)                                                                                                                                       | &#x2714; |
| AugReg        | 2021 | Supervised | ImageNet-22K | [repo](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) | [paper](https://arxiv.org/abs/2106.10270)                                                                                                                                       | -        |
| BEiT          | 2021 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit)                                             | [paper](https://arxiv.org/abs/2106.08254)                                                                                                                                       | -        |
| Uni-Perceiver | 2022 | Supervised | Multi-Modal  | [repo](https://github.com/fundamentalvision/Uni-Perceiver)                                              | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.pdf) | -        |
| BEiTv2        | 2022 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit2)                                            | [paper](https://arxiv.org/abs/2208.06366)                                                                                                                                       | -        |

#### Results and Models

**ADE20K val**

|   Method    |   Backbone    |                                                                                     Pretrain                                                                                     | Lr schd | Crop Size |                                                                                      mIoU (SS/MS)                                                                                       | #Param |                                      Config                                      |                                                                                                                     Download                                                                                                                      | Support? | our mIoU (SS/MS) | our config |
|:-----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|----------|------------------|------------|
|   UperNet   | ViT-Adapter-T |                                                 [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                 |  160k   |    512    |                                                                                       42.6 / 43.6                                                                                       |  36M   |     [config](./configs/ade20k/upernet_deit_adapter_tiny_512_160k_ade20k.py)      |        [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_tiny_512_160_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_deit_adapter_tiny_512_160k_ade20k.log)         | &#x2714; | -/-              |            |
|   UperNet   | ViT-Adapter-S |                                                [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  160k   |    512    |                                                                                       46.2 / 47.1                                                                                       |  58M   |     [config](./configs/ade20k/upernet_deit_adapter_small_512_160k_ade20k.py)     |                                                               [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_small_512_160k_ade20k.pth)                                                               | &#x2714; | 46.09/46.48      |            |
|   UperNet   | ViT-Adapter-B |                                                 [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                 |  160k   |    512    |                                                                                       48.8 / 49.7                                                                                       |  134M  |     [config](./configs/ade20k/upernet_deit_adapter_base_512_160k_ade20k.py)      |        [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_base_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_deit_adapter_base_512_160k_ade20k.log)        | &#x2714; | 48.00/49.21      |            |
|   UperNet   | ViT-Adapter-T | [AugReg-T](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.pth)  |  160k   |    512    |                                                                                       43.9 / 44.8                                                                                       |  36M   |    [config](./configs/ade20k/upernet_augreg_adapter_tiny_512_160k_ade20k.py)     |       [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_tiny_512_160_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_tiny_512_160_ade20k.log)       | &#x2714; | -/-              |            |
|   UperNet   | ViT-Adapter-B | [AugReg-B](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.pth) |  160k   |    512    |                                                                                       51.9 / 52.5                                                                                       |  134M  |    [config](./configs/ade20k/upernet_augreg_adapter_base_512_160k_ade20k.py)     |      [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_base_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_base_512_160k_ade20k.log)      | &#x2714; | -/-              |            |
|   UperNet   | ViT-Adapter-L | [AugReg-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth) |  160k   |    512    |                                                                                       53.4 / 54.4                                                                                       |  364M  |    [config](./configs/ade20k/upernet_augreg_adapter_large_512_160k_ade20k.py)    |     [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_large_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_large_512_160k_ade20k.log)     | &#x2714; | -/-              |            |
|   UperNet   | ViT-Adapter-L |                 [Uni-Perceiver-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/uni-perceiver-large-L24-H1024-224size-pretrained_converted.pth)                 |  160k   |    512    |                                                                                       55.0 / 55.4                                                                                       |  364M  | [config](./configs/ade20k/upernet_uniperceiver_adapter_large_512_160k_ade20k.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_uniperceiver_adapter_large_512_160k_ade20k.pth) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_uniperceiver_adapter_large_512_160k_ade20k.log) | &#x2716; | &#x2716;         |            |
|   UperNet   | ViT-Adapter-L |                              [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)                               |  160k   |    640    | [58.0](https://drive.google.com/file/d/1KsV4QPfoRi5cj2hjCzy8VfWih8xCTrE3/view?usp=sharing) / [58.4](https://drive.google.com/file/d/1haeTUvQhKCM7hunVdK60yxULbRH7YYBK/view?usp=sharing) |  451M  |   [config](./configs/ade20k/upernet_beit_adapter_large_640_160k_ade20k_ss.py)    |     [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.1/upernet_beit_adapter_large_640_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_beit_adapter_large_640_160k_ade20k_ss.log)      | &#x2714; | 58.08/58.16      |            |
| Mask2Former | ViT-Adapter-L |                              [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)                               |  160k   |    640    | [58.3](https://drive.google.com/file/d/1jj56lSbc2s4ZNc-Hi-w6o-OSS99oi-_g/view?usp=sharing) / [59.0](https://drive.google.com/file/d/1hgpZB5gsyd7LTS7Aay2CbHmlY10nafCw/view?usp=sharing) |  568M  | [config](./configs/ade20k/mask2former_beit_adapter_large_640_160k_ade20k_ss.py)  |   [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.2/mask2former_beit_adapter_large_640_160k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_640_160k_ade20k_ss.log)    | &#x2714; | 58.36/-          |            |
| Mask2Former | ViT-Adapter-L |                      [BEiT-L+COCO](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.6/mask2former_beit_adapter_large_896_80k_cocostuff164k.zip)                      |   80k   |    896    | [59.4](https://drive.google.com/file/d/1B_1XSwdnLhjJeUmn1g_nxfvGJpYmYWHa/view?usp=sharing) / [60.5](https://drive.google.com/file/d/1UtjmgcYKR-2h116oQXklUYOVcTw15woM/view?usp=sharing) |  571M  |  [config](./configs/ade20k/mask2former_beit_adapter_large_896_80k_ade20k_ss.py)  |    [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.0/mask2former_beit_adapter_large_896_80k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_896_80k_ade20k_ss.log)     | &#x2714; | -/-              |            |
| Mask2Former | ViT-Adapter-L |                    [BEiTv2-L+COCO](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip)                    |   80k   |    896    |                                                                                       61.2 / 61.5                                                                                       |  571M  | [config](./configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py) |  [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.log)   | &#x2714; | 61.43/-          |            |

Identifier Description:

| Identifier | description                                            |
|------------|--------------------------------------------------------|
| &#x2714;   | Supported                                              |
| &#x2716;   | Not supported, but may be supported in future versions |
| -          | Not tested                                             |

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{chen2022vitadapter,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```