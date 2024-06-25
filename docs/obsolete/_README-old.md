# mmsegExtension

[English](README.md) | [简体中文](README_CN.md)

## Introduction to mmsegExtension

`MmsegExtension` is a robust extension toolbox constructed on the widely used `MMSegmentation` framework.
This toolbox aims to migrate models developed on the `MMSegmentation v0.x` version (like ViT Adapter) to the more
feature-packed and performance-optimized `MMSegmentation v1.x` version.
With `mmsegExtension`, you can directly load the pre-trained weights from `mmseg v0.x` from the original repository and
run and test them seamlessly in the `mmseg v1.x` version.

**Note:** [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main) is an open-source semantic
segmentation toolbox based on PyTorch. It is part of the OpenMMLab project.

## Key Features

- **Code Organization Aligned with `MMSegmentation v1.x`**

  We will organize the file structure according to the directory structure of the MMSegmentation v1.x codebase.
  Developers familiar with MMSegmentation v1.x will be able to get started immediately.

- **Minimal Changes**

  The Config file names remain the same as in the original repository, making it easy for developers familiar with the
  original repository to get started without much hassle.

- **Extensive Pre-trained Weights**

  Directly load pre-trained weights trained by the mmseg v0.x version from the original repository and run and test them
  directly in the mmseg v1.x version.

## Supported repositories

- [ViT-Adapter](https://github.com/czczup/ViT-Adapter)
- [ViT-CoMer](https://github.com/Traffic-X/ViT-CoMer)

### TODO

- [x] Support [ViT-CoMer](https://github.com/Traffic-X/ViT-CoMer) semantic segmentation
- [ ] Support [InternImage](https://github.com/OpenGVLab/InternImage) semantic segmentation
- [x] Support [ViT-Adapter](https://github.com/czczup/ViT-Adapter) semantic segmentation

## Installation and Usage

- **Installation:** Please refer to [get_started.md](docs/readme/get_started.md) for installation.

- **Usage:
  ** [Train and test with existing models](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/4_train_test.md)

- If you are not familiar with `mmseg v1.x`, please read:
    - [Getting started with MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/15_minutes.html)
    - [Overview of MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/overview.md)

## Completed work

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
| InternImage-H  |   UperNet   |  896x896   | 59.9 / 60.3  | 1.12B  | 3566G |               [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_h_896_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_internimage_h_896_160k_ade20k.py)                | &#x2714; | 59.49/-              | [config](./configs/internimage/upernet_internimage_h_512_160k_ade20k.py)  |
| InternImage-H  | Mask2Former |  896x896   | 62.5 / 62.9  | 1.31B  | 4635G | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth) \| [cfg](segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py) | &#x2716; | -/-              |                                                                           |

</div>

</details>

## Why are the test results inconsistent with the original repository?

1. **Changes in PyTorch version**

   New versions of PyTorch may introduce new features and changes that may affect the training and testing results of
   your model.
   For example, PyTorch 1.7 introduced support for CUDA11, which may affect the performance and stability of training
   and testing using GPUs.
   The PyTorch 2.0 version brings new compilers and performance optimizations such as torch.compile, which may change
   the execution method and results of models.

2. **Changes in CUDA version**

   CUDA is NVIDIA's GPU-accelerated computing platform, and PyTorch uses CUDA for GPU-accelerated training and testing.
   When the CUDA version is updated, it may affect the performance and stability of GPU training and testing.
   For example, upgrading to CUDA 11 may affect the performance of PyTorch 1.7 and above.