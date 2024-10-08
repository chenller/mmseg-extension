
# Get started: Install and Run mmsegextension

## Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMSegmentation works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

**MMsegExtension depends on MMSegmentation v1.x.** 

**Note:**
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](##installation). Otherwise, you can follow these steps for the preparation.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name mmsegext python=3.9 -y
conda activate mmsegext
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
# Example
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```


## Installation

We recommend that users follow our best practices to install MMSegmentation. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
```

**Step 1.** Install `MMSegmentation` and `MMDetection`.

```shell
pip install "mmsegmentation>=1.0.0"
pip install mmdet
```
**Step 2.** Install `mmseg-extension` and `mmseg-extension-lib`

```shell
git clone https://github.com/chenller/mmseg-extension.git
cd mmseg-extension
bash install.sh
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
**Step 3.** Install requirements

```shell
pip install timm fairscale einops ftfy regex 
pip install "opencv-python<=4.9.0"
```