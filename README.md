# mmseg-extension

[English](README.md) | [简体中文](README_CN.md)

## Introduction

`mmseg-extension` is a comprehensive extension of
the [MMSegmentation library (version 1.x)](https://github.com/open-mmlab/mmsegmentation/tree/main),
designed to provide a more versatile and up-to-date framework for semantic segmentation.
This repository consolidates the latest advancements in semantic segmentation
by integrating and unifying various models and codes within the MMSegmentation ecosystem.
Users benefit from a consistent and streamlined training and testing process,
significantly reducing the learning curve and enhancing productivity.

The main branch works with PyTorch 2.0 or higher (we recommend PyTorch 2.3).
You can still use PyTorch 1.x, but no testing has been conducted.

## Features and Objectives

- **MMSegmentation Extension**

  This repository extends the capabilities of MMSegmentation 1.x,
  leveraging its robust framework for semantic segmentation tasks.

- **Model Migration**

  Models from MMSegmentation 0.x are migrated to be compatible with MMSegmentation 1.x.

- **Integration of External Codes**

  Codes and models not originally developed with MMSegmentation can be adapted to
  use MMSegmentation's data loading, training, and validation mechanisms.

- **Model Weights Compatibility**

  Models trained in their original repositories can be used directly for training and inference
  in mmseg-extension without the need for retraining.

- **Tracking Latest Models**

  The repository stays updated with the latest research and models in semantic segmentation.

- **Minimal Changes**

  The Config file names remain the same as in the original repository, making it easy for developers familiar with the
  original repository to get started without much hassle.

<details>
<summary> Addressing Key Issues </summary>
<br>
<div>

- **Staying Current with Latest Models**

  mmseg-extension addresses the delay in MMSegmentation's inclusion of the latest models by continuously integrating the
  newest research.

- **Standardizing Disparate Codebases**

  By providing a unified framework, mmseg-extension solves the problem of inconsistent data loading, training, and
  validation scripts across different research papers.

- **Utilizing Pre-trained Weights**

  Ensures compatibility with pre-trained weights from various repositories, enabling seamless model integration without
  the need for retraining.

</div>

</details>

## Installation and Usage

- **Installation:** Please refer to [get_started.md](docs/readme/get_started.md) for installation.

- **Usage:**
  [Train and test with existing models](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/4_train_test.md)

- If you are not familiar with `mmseg v1.x`, please read:
    - [Getting started with MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/15_minutes.html)
    - [Overview of MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/overview.md)

## Overview of Model Zoo

| Name        | Year | Publication | Paper                                     | Code                                                                       |
|-------------|------|-------------|-------------------------------------------|----------------------------------------------------------------------------|
| ViT-Adapter | 2023 | ICLR        | [Arxiv](https://arxiv.org/abs/2205.08534) | [Code](https://github.com/czczup/ViT-Adapter)                              |
| ViT-CoMer   | 2024 | CVPR        | [Arxiv](https://arxiv.org/abs/2403.07392) | [Code](https://github.com/Traffic-X/ViT-CoMer)                             |
| InternImage | 2023 | CVPR        | [Arxiv](https://arxiv.org/abs/2211.05778) | [Code](https://github.com/OpenGVLab/InternImage/tree/master/segmentation)  |
| TransNeXt   | 2024 | CVPR        | [Arxiv](https://arxiv.org/abs/2311.17132) | [Code](https://github.com/DaiShiResearch/TransNeXt/tree/main/segmentation) |
| UniRepLKNet | 2024 | CVPR        | [Arxiv](https://arxiv.org/abs/2311.15599) | [Code](https://github.com/ailab-cvc/unireplknet)                           |

### Completed Work Results

<details>
<summary> Identifier Description </summary>
<br>
<div>

| Identifier | description                                            |
|------------|--------------------------------------------------------|
| &#x2714;   | Supported                                              |
| &#x2716;   | Not supported, but may be supported in future versions |
| **-**      | Not tested                                             |

</div>

</details>

### [ViT-Adapter](https://github.com/czczup/ViT-Adapter)

> You can find detailed information about ViT Adapters
> in [README.md](https://github.com/czczup/ViT-Adapter/blob/main/segmentation/README.md).

<details>
<summary> ViT-Adapter Pretraining Sources </summary>
<br>
<div>

| Name          | Year | Type       | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           | Support? |
|---------------|------|------------|--------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| DeiT          | 2021 | Supervised | ImageNet-1K  | [repo](https://github.com/facebookresearch/deit/blob/main/README_deit.md)                               | [paper](https://arxiv.org/abs/2012.12877)                                                                                                                                       | &#x2714; |
| AugReg        | 2021 | Supervised | ImageNet-22K | [repo](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) | [paper](https://arxiv.org/abs/2106.10270)                                                                                                                                       | -        |
| BEiT          | 2021 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit)                                             | [paper](https://arxiv.org/abs/2106.08254)                                                                                                                                       | -        |
| Uni-Perceiver | 2022 | Supervised | Multi-Modal  | [repo](https://github.com/fundamentalvision/Uni-Perceiver)                                              | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Uni-Perceiver_Pre-Training_Unified_Architecture_for_Generic_Perception_for_Zero-Shot_and_CVPR_2022_paper.pdf) | &#x2716; |
| BEiTv2        | 2022 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit2)                                            | [paper](https://arxiv.org/abs/2208.06366)                                                                                                                                       | -        |

</div>

</details>


<details>
<summary> ViT-Adapter ADE20K val </summary>
<br>
<div>

|   Method    |   Backbone    |                                                                                     Pretrain                                                                                     | Lr schd | Crop Size |                                                                                      mIoU (SS/MS)                                                                                       | #Param |                                      Config                                      |                                                                                                                     Download                                                                                                                      | Support? | our mIoU (SS/MS) | our config                                                                            |
|:-----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|----------|------------------|---------------------------------------------------------------------------------------|
|   UperNet   | ViT-Adapter-T |                                                 [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                 |  160k   |    512    |                                                                                       42.6 / 43.6                                                                                       |  36M   |     [config](./configs/ade20k/upernet_deit_adapter_tiny_512_160k_ade20k.py)      |        [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_tiny_512_160_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_deit_adapter_tiny_512_160k_ade20k.log)         | &#x2714; | -/-              | [config](./configs/vit_adapter/upernet_deit_adapter_tiny_512_160k_ade20k.py)          |
|   UperNet   | ViT-Adapter-S |                                                [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  160k   |    512    |                                                                                       46.2 / 47.1                                                                                       |  58M   |     [config](./configs/ade20k/upernet_deit_adapter_small_512_160k_ade20k.py)     |                                                               [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_small_512_160k_ade20k.pth)                                                               | &#x2714; | 46.09/46.48      | [config](./configs/vit_adapter/upernet_deit_adapter_small_512_160k_ade20k.py)         |
|   UperNet   | ViT-Adapter-B |                                                 [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                 |  160k   |    512    |                                                                                       48.8 / 49.7                                                                                       |  134M  |     [config](./configs/ade20k/upernet_deit_adapter_base_512_160k_ade20k.py)      |        [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_base_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_deit_adapter_base_512_160k_ade20k.log)        | &#x2714; | 48.00/49.21      | [config](./configs/vit_adapter/upernet_deit_adapter_base_512_160k_ade20k.py)          |
|   UperNet   | ViT-Adapter-T | [AugReg-T](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.pth)  |  160k   |    512    |                                                                                       43.9 / 44.8                                                                                       |  36M   |    [config](./configs/ade20k/upernet_augreg_adapter_tiny_512_160k_ade20k.py)     |       [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_tiny_512_160_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_tiny_512_160_ade20k.log)       | &#x2714; | -/-              | [config](./configs/vit_adapter/upernet_augreg_adapter_tiny_512_160k_ade20k.py)        |
|   UperNet   | ViT-Adapter-B | [AugReg-B](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.pth) |  160k   |    512    |                                                                                       51.9 / 52.5                                                                                       |  134M  |    [config](./configs/ade20k/upernet_augreg_adapter_base_512_160k_ade20k.py)     |      [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_base_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_base_512_160k_ade20k.log)      | &#x2714; | -/-              | [config](./configs/vit_adapter/upernet_augreg_adapter_base_512_160k_ade20k.py)        |
|   UperNet   | ViT-Adapter-L | [AugReg-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth) |  160k   |    512    |                                                                                       53.4 / 54.4                                                                                       |  364M  |    [config](./configs/ade20k/upernet_augreg_adapter_large_512_160k_ade20k.py)    |     [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_large_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_large_512_160k_ade20k.log)     | &#x2714; | -/-              | [config](./configs/vit_adapter/upernet_augreg_adapter_large_512_160k_ade20k.py)       |
|   UperNet   | ViT-Adapter-L |                 [Uni-Perceiver-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/uni-perceiver-large-L24-H1024-224size-pretrained_converted.pth)                 |  160k   |    512    |                                                                                       55.0 / 55.4                                                                                       |  364M  | [config](./configs/ade20k/upernet_uniperceiver_adapter_large_512_160k_ade20k.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_uniperceiver_adapter_large_512_160k_ade20k.pth) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_uniperceiver_adapter_large_512_160k_ade20k.log) | &#x2716; | &#x2716;         | &#x2716;                                                                              |
|   UperNet   | ViT-Adapter-L |                              [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)                               |  160k   |    640    | [58.0](https://drive.google.com/file/d/1KsV4QPfoRi5cj2hjCzy8VfWih8xCTrE3/view?usp=sharing) / [58.4](https://drive.google.com/file/d/1haeTUvQhKCM7hunVdK60yxULbRH7YYBK/view?usp=sharing) |  451M  |   [config](./configs/ade20k/upernet_beit_adapter_large_640_160k_ade20k_ss.py)    |     [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.1/upernet_beit_adapter_large_640_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_beit_adapter_large_640_160k_ade20k_ss.log)      | &#x2714; | 58.08/58.16      | [config](./configs/vit_adapter/upernet_beit_adapter_large_640_160k_ade20k_ss.py)      |
| Mask2Former | ViT-Adapter-L |                              [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)                               |  160k   |    640    | [58.3](https://drive.google.com/file/d/1jj56lSbc2s4ZNc-Hi-w6o-OSS99oi-_g/view?usp=sharing) / [59.0](https://drive.google.com/file/d/1hgpZB5gsyd7LTS7Aay2CbHmlY10nafCw/view?usp=sharing) |  568M  | [config](./configs/ade20k/mask2former_beit_adapter_large_640_160k_ade20k_ss.py)  |   [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.2/mask2former_beit_adapter_large_640_160k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_640_160k_ade20k_ss.log)    | &#x2714; | 58.36/-          | [config](./configs/vit_adapter/mask2former_beit_adapter_large_640_160k_ade20k_ss.py)  |
| Mask2Former | ViT-Adapter-L |                      [BEiT-L+COCO](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.6/mask2former_beit_adapter_large_896_80k_cocostuff164k.zip)                      |   80k   |    896    | [59.4](https://drive.google.com/file/d/1B_1XSwdnLhjJeUmn1g_nxfvGJpYmYWHa/view?usp=sharing) / [60.5](https://drive.google.com/file/d/1UtjmgcYKR-2h116oQXklUYOVcTw15woM/view?usp=sharing) |  571M  |  [config](./configs/ade20k/mask2former_beit_adapter_large_896_80k_ade20k_ss.py)  |    [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.0/mask2former_beit_adapter_large_896_80k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_896_80k_ade20k_ss.log)     | &#x2714; | -/-              | [config](./configs/vit_adapter/mask2former_beit_adapter_large_896_80k_ade20k_ss.py)   |
| Mask2Former | ViT-Adapter-L |                    [BEiTv2-L+COCO](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip)                    |   80k   |    896    |                                                                                       61.2 / 61.5                                                                                       |  571M  | [config](./configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py) |  [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.log)   | &#x2714; | 61.43/-          | [config](./configs/vit_adapter/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py) |

</div>

</details>

### [ViT-CoMer](https://github.com/Traffic-X/ViT-CoMer)

<details>
<summary> ViT-CoMer ADE20K val </summary>
<br>
<div>

| Method  |  Backbone   |                              Pretrain                              | Lr schd | Crop Size | mIoU(SS/MS) | #Param |                               Config                               |                               Ckpt                               |                               Log                               | Support? | our mIoU (SS/MS) | our config                                                               |
|:-------:|:-----------:|:------------------------------------------------------------------:|:-------:|:---------:|:-----------:|:------:|:------------------------------------------------------------------:|:----------------------------------------------------------------:|:---------------------------------------------------------------:|----------|------------------|--------------------------------------------------------------------------|
| UperNet | ViT-CoMer-T | [DeiT-T](https://pan.baidu.com/s/1684XaK4dRb8crxb8DRrQ7Q?pwd=fxqa) |  160k   |    512    |   43.5/-    | 38.7M  | [config](https://pan.baidu.com/s/1KxzkLZu8qXi9wfIe3JF04w?pwd=4gjs) | [ckpt](https://pan.baidu.com/s/1J_XgJ058PpK8gqz9E0Caig?pwd=k6mf) | [log](https://pan.baidu.com/s/1qh6xvubnU9Y6bG6UNp22IA?pwd=3p8u) | &#x2714; | 43.66/-          | [config](./configs/vit_comer/upernet_vit_comer_tiny_512_160k_ade20k.py)  |
| UperNet | ViT-CoMer-S | [DeiT-S](https://pan.baidu.com/s/1HCvcilNKPgCp4gYbsSLQpw?pwd=p4jg) |  160k   |    512    |   46.5/-    | 61.4M  | [config](https://pan.baidu.com/s/1H3PC01bMQvquRLvd4JHuuA?pwd=kgyy) | [ckpt](https://pan.baidu.com/s/1CDfKeUzCTs5fB0ggy9wYwg?pwd=puqi) | [log](https://pan.baidu.com/s/1nci50aHO0ma3YgIzH-z9NQ?pwd=cxdj) | &#x2714; | 46.09/46.23      | [config](./configs/vit_comer/upernet_vit_comer_small_512_160k_ade20k.py) |
| UperNet | ViT-CoMer-B | [DeiT-S](https://pan.baidu.com/s/1XuTrT95i1XC52bzYeFdIQw?pwd=9kab) |  160k   |    512    |   48.8/-    | 144.7M |                                 -                                  |                                -                                 |                                -                                | &#x2714; | -/-              | [config](./configs/vit_comer/upernet_vit_comer_base_512_160k_ade20k.py)  |

</div>

</details>

### [InternImage](https://github.com/OpenGVLab/InternImage)

<details>
<summary> InternImage ADE20K Semantic Segmentation </summary>
<br>
<div>

|    backbone    |   method    | resolution | mIoU (ss/ms) | #param | FLOPs |                                                                                                        download                                                                                                         | Support? | our mIoU (SS/MS) | our config                                                                |
|:--------------:|:-----------:|:----------:|:------------:|:------:|:-----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|----------|------------------|---------------------------------------------------------------------------|
| InternImage-T  |   UperNet   |  512x512   | 47.9 / 48.1  |  59M   | 944G  |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_t_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k.py)                | &#x2714; | 47.60/-          | [config](./configs/internimage/upernet_internimage_t_512_160k_ade20k.py)  |
| InternImage-S  |   UperNet   |  512x512   | 50.1 / 50.9  |  80M   | 1017G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_s_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_s_512_160k_ade20k.py)                | &#x2714; | 49.77/-          | [config](./configs/internimage/upernet_internimage_s_512_160k_ade20k.py)  |
| InternImage-B  |   UperNet   |  512x512   | 50.8 / 51.3  |  128M  | 1185G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_b_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_b_512_160k_ade20k.py)                | &#x2714; | 50.46/51.05      | [config](./configs/internimage/upernet_internimage_b_512_160k_ade20k.py)  |
| InternImage-L  |   UperNet   |  640x640   | 53.9 / 54.1  |  256M  | 2526G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_l_640_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_l_640_160k_ade20k.py)                | &#x2714; | 53.39/-          | [config](./configs/internimage/upernet_internimage_l_512_160k_ade20k.py)  |
| InternImage-XL |   UperNet   |  640x640   | 55.0 / 55.3  |  368M  | 3142G |              [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_xl_640_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_xl_640_160k_ade20k.py)               | &#x2714; | 54.4/-           | [config](./configs/internimage/upernet_internimage_xl_512_160k_ade20k.py) |
| InternImage-H  |   UperNet   |  896x896   | 59.9 / 60.3  | 1.12B  | 3566G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_h_896_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_h_896_160k_ade20k.py)                | &#x2714; | 59.49/-          | [config](./configs/internimage/upernet_internimage_h_512_160k_ade20k.py)  |
| InternImage-H  | Mask2Former |  896x896   | 62.5 / 62.9  | 1.31B  | 4635G | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth) \| [cfg](segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py) | &#x2716; | -/-              |                                                                           |

</div>
</details>

### [TransNeXt](https://github.com/DaiShiResearch/TransNeXt/tree/main/segmentation)

<details>
<summary> TransNeXt ADE20K Semantic Segmentation using the UPerNet method </summary>
<br>
<div>

|    Backbone     |                                                         Pretrained Model                                                          | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #Params |                                                                          Download                                                                          |                                          Config                                           |                                                                    Log                                                                     | Support? | our mIoU (SS/MS) | our config                                                                      |
|:---------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------:|:-------:|:----:|:--------------:|:-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|----------|------------------|---------------------------------------------------------------------------------|
| TransNeXt-Tiny  |  [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)  |  512x512  |  160K   | 51.1 |   51.5/51.7    |   59M   |  [model](https://huggingface.co/DaiShiResearch/upernet-transnext-tiny-ade/resolve/main/upernet_transnext_tiny_512x512_160k_ade20k_in1k.pth?download=true)  | [config](/segmentation/upernet/configs/upernet_transnext_tiny_512x512_160k_ade20k_ss.py)  |  [log](https://huggingface.co/DaiShiResearch/upernet-transnext-tiny-ade/blob/main/upernet_transnext_tiny_512x512_160k_ade20k_ss.log.json)  | &#x2714; | 53.02/-          | [config](./configs/transnext/upernet_transnext_base_512x512_160k_ade20k_ss.py)  |
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true) |  512x512  |  160K   | 52.2 |   52.5/52.8    |   80M   | [model](https://huggingface.co/DaiShiResearch/upernet-transnext-small-ade/resolve/main/upernet_transnext_small_512x512_160k_ade20k_in1k.pth?download=true) | [config](/segmentation/upernet/configs/upernet_transnext_small_512x512_160k_ade20k_ss.py) | [log](https://huggingface.co/DaiShiResearch/upernet-transnext-small-ade/blob/main/upernet_transnext_small_512x512_160k_ade20k_ss.log.json) | &#x2714; | 52.15/-          | [config](./configs/transnext/upernet_transnext_small_512x512_160k_ade20k_ss.py) |
| TransNeXt-Base  |  [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true)  |  512x512  |  160K   | 53.0 |   53.5/53.7    |  121M   |  [model](https://huggingface.co/DaiShiResearch/upernet-transnext-base-ade/resolve/main/upernet_transnext_base_512x512_160k_ade20k_in1k.pth?download=true)  | [config](/segmentation/upernet/configs/upernet_transnext_base_512x512_160k_ade20k_ss.py)  |  [log](https://huggingface.co/DaiShiResearch/upernet-transnext-base-ade/blob/main/upernet_transnext_base_512x512_160k_ade20k_ss.log.json)  | &#x2714; | 51.11/-          | [config](./configs/transnext/upernet_transnext_tiny_512x512_160k_ade20k_ss.py)  |

* In the context of multi-scale evaluation, TransNeXt reports test results under two distinct scenarios:
  **interpolation** and **extrapolation** of relative position bias.

</div>
</details>

<details>
<summary> TransNeXt ADE20K Semantic Segmentation using the Mask2Former method </summary>
<br>
<div>

|    Backbone     |                                                         Pretrained Model                                                          | Crop Size | Lr Schd | mIoU | #Params |                                                                              Download                                                                              |                                             Config                                             |                                                                       Log                                                                       | Support? | our mIoU (SS/MS) | our config                                                                       |
|:---------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:---------:|:-------:|:----:|:-------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|----------|------------------|----------------------------------------------------------------------------------|
| TransNeXt-Tiny  |  [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)  |  512x512  |  160K   | 53.4 |  47.5M  |  [model](https://huggingface.co/DaiShiResearch/mask2former-transnext-tiny-ade/resolve/main/mask2former_transnext_tiny_512x512_160k_ade20k_in1k.pth?download=true)  | [config](/segmentation/mask2former/configs/mask2former_transnext_tiny_160k_ade20k-512x512.py)  |  [log](https://huggingface.co/DaiShiResearch/mask2former-transnext-tiny-ade/raw/main/mask2former_transnext_tiny_512x512_160k_ade20k_in1k.json)  | &#x2714; | 53.43/-          | [config](./configs/transnext/mask2former_transnext_base_160k_ade20k-512x512.py)  |
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true) |  512x512  |  160K   | 54.1 |  69.0M  | [model](https://huggingface.co/DaiShiResearch/mask2former-transnext-small-ade/resolve/main/mask2former_transnext_small_512x512_160k_ade20k_in1k.pth?download=true) | [config](/segmentation/mask2former/configs/mask2former_transnext_small_160k_ade20k-512x512.py) | [log](https://huggingface.co/DaiShiResearch/mask2former-transnext-small-ade/raw/main/mask2former_transnext_small_512x512_160k_ade20k_in1k.json) | &#x2714; | 54.06/-          | [config](./configs/transnext/mask2former_transnext_small_160k_ade20k-512x512.py) |
| TransNeXt-Base  |  [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true)  |  512x512  |  160K   | 54.7 |  109M   |  [model](https://huggingface.co/DaiShiResearch/mask2former-transnext-base-ade/resolve/main/mask2former_transnext_base_512x512_160k_ade20k_in1k.pth?download=true)  | [config](/segmentation/mask2former/configs/mask2former_transnext_base_160k_ade20k-512x512.py)  |  [log](https://huggingface.co/DaiShiResearch/mask2former-transnext-base-ade/raw/main/mask2former_transnext_base_512x512_160k_ade20k_in1k.json)  | &#x2714; | 54.68/-          | [config](./configs/transnext/mask2former_transnext_tiny_160k_ade20k-512x512.py)  |

</div>
</details>

### [UniRepLKNet](https://github.com/ailab-cvc/unireplknet)

<details>
<summary> UniRepLKNet ADE20K Semantic Segmentation </summary>
<br>
<div>

|        name        | resolution | mIoU (ss/ms) | #params | FLOPs |                                            Weights                                            | Support? | our mIoU (SS/MS) | our config                                                                     |
|:------------------:|:----------:|:------------:|:-------:|:-----:|:---------------------------------------------------------------------------------------------:|----------|------------------|--------------------------------------------------------------------------------|
|   UniRepLKNet-T    |  512x512   |  48.6/49.1   |   61M   | 946G  | [ckpt](https://drive.google.com/file/d/1R2teeQt7q48EBBRbeVXShISpOmS5YHjs/view?usp=drive_link) | &#x2714; | 47.94/-          | [config](./configs/unireplknet/upernet_unireplknet_t_512_160k_ade20k.py)       |
|   UniRepLKNet-S    |  512x512   |  50.5/51.0   |   86M   | 1036G | [ckpt](https://drive.google.com/file/d/1SBHvbK4zoPSZ827F5Sp209LYIh2T7Iew/view?usp=drive_link) | &#x2714; | -/-              | [config](./configs/unireplknet/upernet_unireplknet_s_512_160k_ade20k.py)       |
| UniRepLKNet-S_22K  |  512x512   |  51.9/52.7   |   86M   | 1036G | [ckpt](https://drive.google.com/file/d/15dNuw34kia5qtt6UijcnutEktY05OrKH/view?usp=drive_link) | &#x2714; | -/-              | [config](./configs/unireplknet/upernet_unireplknet_s_in22k_512_160k_ade20k.py) |
| UniRepLKNet-S_22K  |  640x640   |  52.3/52.7   |   86M   | 1618G |  [ckpt](https://drive.google.com/file/d/1WVmAQ8sKDeX0APS9Q88z4dZge31kHx2v/view?usp=sharing)   | &#x2714; | -/-              | [config](./configs/unireplknet/upernet_unireplknet_s_in22k_640_160k_ade20k.py) |   
| UniRepLKNet-B_22K  |  640x640   |  53.5/53.9   |  130M   | 1850G | [ckpt](https://drive.google.com/file/d/1sflCn8ny-cU5Bk8yBGE3E-yIO8eECE0H/view?usp=drive_link) | &#x2714; | 52.89/-          | [config](./configs/unireplknet/upernet_unireplknet_b_in22k_640_160k_ade20k.py) |
| UniRepLKNet-L_22K  |  640x640   |  54.5/55.0   |  254M   | 2507G | [ckpt](https://drive.google.com/file/d/1Qev75aKZY5bNAM17cLecD2OoZwKf5DA7/view?usp=drive_link) | &#x2714; | -/-              | [config](./configs/unireplknet/upernet_unireplknet_l_in22k_640_160k_ade20k.py) |
| UniRepLKNet-XL_22K |  640x640   |  55.2/55.6   |  425M   | 3420G |  [ckpt](https://drive.google.com/file/d/1Ajwc7ZOk5eK19XX6VzgmAu2Wn0Dkb3jI/view?usp=sharing)   | &#x2716; | -/-              | -                                                                              |

**NOTE:** Checkpoints have already been released on hugging face. You can download them right now
from https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main.

</div>
</details>

