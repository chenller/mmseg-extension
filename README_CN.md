# mmsegExtension

[English](README.md) | [简体中文](README_CN.md)

## mmsegExtension简介

mmsegExtension是一个功能强大的扩展工具箱，它建立在广受欢迎的MMSegmentation框架之上。
这个工具箱将基于 **MMSegmentation v0.x 版本**开发的模型（例如ViT-Adapter）迁移到功能更丰富、性能更优化的 **MMSegmentation
v1.x版本**。
通过mmsegExtension，直接加载原始仓库由**mmseg v0.x**版本训练的预训练权重，直接在**mmseg v1.x**版本运行与测试。

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main)是一个基于 PyTorch 的语义分割开源工具箱。它是
OpenMMLab 项目的一部分。

### 主要特性

- **代码组织与MMSegmentation v1.x相同**

  我们将按照MMSegmentation v1.x代码库的目录结构组织文件结构。熟悉MMSegmentation v1.x的开发人员将能直接上手。

- **最小的更改**

  Config配置名称与原存储库相同，熟悉原存储库的开发者能直接上手。

- **丰富的预训练权重**

  直接加载原始仓库由**mmseg v0.x**版本训练的预训练权重，直接在**mmseg v1.x**版本运行与测试。

## 支持的存储库

- [ViT-Adapter](https://github.com/czczup/ViT-Adapter)

## TODO

- [ ] Support [ViT-CoMer](https://github.com/Traffic-X/ViT-CoMer) semantic segmentation
- [ ] Support [InternImage](https://github.com/OpenGVLab/InternImage) semantic segmentation
- [x] Support [ViT-Adapter](https://github.com/czczup/ViT-Adapter) semantic segmentation

## 安装与使用

**安装:** 请参阅 [get_started.md](docs/readme/get_started.md) 进行安装。

**使用:
** [使用现有模型进行训练和测试](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/4_train_test.md)

> 如果您不熟悉`mmseg v1.x`，请阅读：
>  - [Getting started with MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/15_minutes.html)
>  - [Overview of MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/overview.md)

## 已完成的工作

<details>
<summary> 符号说明 </summary>
<br>
<div>

| 符号       | 描述                |
|----------|-------------------|
| &#x2714; | 已支持               |
| &#x2716; | 未支持，但在未来的版本可能将会支持 |
| **-**    | 未测试               |

</div>
</details>

### [ViT-Adapter](https://github.com/czczup/ViT-Adapter)

> 你能在[README.md](https://github.com/czczup/ViT-Adapter/blob/main/segmentation/README.md)找到关于ViT-Adapter的详细资料

<details>
<summary> ViT-Adapter 的预训练模型 </summary>
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
<summary> ViT-Adapter 在 ADE20K val 数据集的预选连权重 </summary>
<br>
<div>

|   Method    |   Backbone    |                                                                                     Pretrain                                                                                     | Lr schd | Crop Size |                                                                                      mIoU (SS/MS)                                                                                       | #Param |                                      Config                                      |                                                                                                                     Download                                                                                                                      | Support? | our mIoU (SS/MS) | our config                                                                            |
|:-----------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|----------|------------------|---------------------------------------------------------------------------------------|
|   UperNet   | ViT-Adapter-T |                                                 [DeiT-T](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)                                                 |  160k   |    512    |                                                                                       42.6 / 43.6                                                                                       |  36M   |     [config](./configs/ade20k/upernet_deit_adapter_tiny_512_160k_ade20k.py)      |        [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_tiny_512_160_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_deit_adapter_tiny_512_160k_ade20k.log)         | &#x2714; | -/-              | **-**                                                                                 |
|   UperNet   | ViT-Adapter-S |                                                [DeiT-S](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)                                                 |  160k   |    512    |                                                                                       46.2 / 47.1                                                                                       |  58M   |     [config](./configs/ade20k/upernet_deit_adapter_small_512_160k_ade20k.py)     |                                                               [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_small_512_160k_ade20k.pth)                                                               | &#x2714; | 46.09/46.48      | [config](./configs/vit_adapter/upernet_deit_adapter_small_512_160k_ade20k.py)         |
|   UperNet   | ViT-Adapter-B |                                                 [DeiT-B](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth)                                                 |  160k   |    512    |                                                                                       48.8 / 49.7                                                                                       |  134M  |     [config](./configs/ade20k/upernet_deit_adapter_base_512_160k_ade20k.py)      |        [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_deit_adapter_base_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_deit_adapter_base_512_160k_ade20k.log)        | &#x2714; | 48.00/49.21      | [config](./configs/vit_adapter/upernet_deit_adapter_base_512_160k_ade20k.py)          |
|   UperNet   | ViT-Adapter-T | [AugReg-T](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.pth)  |  160k   |    512    |                                                                                       43.9 / 44.8                                                                                       |  36M   |    [config](./configs/ade20k/upernet_augreg_adapter_tiny_512_160k_ade20k.py)     |       [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_tiny_512_160_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_tiny_512_160_ade20k.log)       | &#x2714; | -/-              | [config](./configs/vit_adapter/upernet_augreg_adapter_tiny_512_160k_ade20k.py)        |
|   UperNet   | ViT-Adapter-B | [AugReg-B](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.pth) |  160k   |    512    |                                                                                       51.9 / 52.5                                                                                       |  134M  |    [config](./configs/ade20k/upernet_augreg_adapter_base_512_160k_ade20k.py)     |      [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_base_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_base_512_160k_ade20k.log)      | &#x2714; | -/-              | [config](./configs/vit_adapter/upernet_augreg_adapter_base_512_160k_ade20k.py)        |
|   UperNet   | ViT-Adapter-L | [AugReg-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth) |  160k   |    512    |                                                                                       53.4 / 54.4                                                                                       |  364M  |    [config](./configs/ade20k/upernet_augreg_adapter_large_512_160k_ade20k.py)    |     [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_augreg_adapter_large_512_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_augreg_adapter_large_512_160k_ade20k.log)     | &#x2714; | -/-              | [config](./configs/vit_adapter/upernet_augreg_adapter_large_512_160k_ade20k.py)       |
|   UperNet   | ViT-Adapter-L |                 [Uni-Perceiver-L](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/uni-perceiver-large-L24-H1024-224size-pretrained_converted.pth)                 |  160k   |    512    |                                                                                       55.0 / 55.4                                                                                       |  364M  | [config](./configs/ade20k/upernet_uniperceiver_adapter_large_512_160k_ade20k.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/upernet_uniperceiver_adapter_large_512_160k_ade20k.pth) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_uniperceiver_adapter_large_512_160k_ade20k.log) | &#x2716; | &#x2716;         | &#x2716;                                                                              |
|   UperNet   | ViT-Adapter-L |                              [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)                               |  160k   |    640    | [58.0](https://drive.google.com/file/d/1KsV4QPfoRi5cj2hjCzy8VfWih8xCTrE3/view?usp=sharing) / [58.4](https://drive.google.com/file/d/1haeTUvQhKCM7hunVdK60yxULbRH7YYBK/view?usp=sharing) |  451M  |   [config](./configs/ade20k/upernet_beit_adapter_large_640_160k_ade20k_ss.py)    |     [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.1/upernet_beit_adapter_large_640_160k_ade20k.pth.tar) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/upernet_beit_adapter_large_640_160k_ade20k_ss.log)      | &#x2714; | 58.08/58.16      | [config](./configs/vit_adapter/upernet_beit_adapter_large_640_160k_ade20k_ss.py)      |
| Mask2Former | ViT-Adapter-L |                              [BEiT-L](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k_ft22k.pth)                               |  160k   |    640    | [58.3](https://drive.google.com/file/d/1jj56lSbc2s4ZNc-Hi-w6o-OSS99oi-_g/view?usp=sharing) / [59.0](https://drive.google.com/file/d/1hgpZB5gsyd7LTS7Aay2CbHmlY10nafCw/view?usp=sharing) |  568M  | [config](./configs/ade20k/mask2former_beit_adapter_large_640_160k_ade20k_ss.py)  |   [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.2/mask2former_beit_adapter_large_640_160k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_640_160k_ade20k_ss.log)    | &#x2714; | 58.36/-          | [config](./configs/vit_adapter/mask2former_beit_adapter_large_640_160k_ade20k_ss.py)  |
| Mask2Former | ViT-Adapter-L |                      [BEiT-L+COCO](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.6/mask2former_beit_adapter_large_896_80k_cocostuff164k.zip)                      |   80k   |    896    | [59.4](https://drive.google.com/file/d/1B_1XSwdnLhjJeUmn1g_nxfvGJpYmYWHa/view?usp=sharing) / [60.5](https://drive.google.com/file/d/1UtjmgcYKR-2h116oQXklUYOVcTw15woM/view?usp=sharing) |  571M  |  [config](./configs/ade20k/mask2former_beit_adapter_large_896_80k_ade20k_ss.py)  |    [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.2.0/mask2former_beit_adapter_large_896_80k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_896_80k_ade20k_ss.log)     | &#x2714; | -/-              | **-**                                                                                 |
| Mask2Former | ViT-Adapter-L |                    [BEiTv2-L+COCO](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip)                    |   80k   |    896    |                                                                                       61.2 / 61.5                                                                                       |  571M  | [config](./configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py) |  [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_ade20k.zip) \| [log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.log)   | &#x2714; | 61.43/-          | [config](./configs/vit_adapter/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py) |

</div>

</details>

## 为什么测试结果和原存储库不一致？

1. PyTorch版本的变化

   PyTorch的新版本可能会引入新的功能和更改，这些更改可能会影响模型的训练和测试结果。
   例如，PyTorch 1.7版本中引入了CUDA11的支持，这可能会影响使用GPU进行训练和测试的性能和稳定性。
   PyTorch 2.0版本则带来了API的torch.compile等新的编译器和性能优化，这些更新可能改变了模型的执行方式和结果。

2. CUDA版本的变化：

   CUDA是NVIDIA的GPU加速计算平台，PyTorch使用CUDA进行GPU加速训练和测试。
   当CUDA版本更新时，可能会影响GPU训练和测试的性能和稳定性。
   例如，升级到CUDA11可能会影响PyTorch 1.7及以上版本的性能。


