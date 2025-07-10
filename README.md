
# CycleGAN++ in PyTorch

This repository contains a PyTorch implementation for unpaired image-to-image translation using the CycleGAN++ model.

This implementation builds upon the original CycleGAN work.

**CycleGAN: [Paper](https://arxiv.org/pdf/1703.10593.pdf)**

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGANPlusPlus
cd pytorch-CycleGANPlusPlus
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGANPlusPlus)](https://repl.it/github/junyanz/pytorch-CycleGANPlusPlus).

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. horse2zebra):
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/horse2zebra/ --name horse2zebra --model cycle_gan_plusplus
```
To see more intermediate results, check out `./checkpoints/horse2zebra/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/horse2zebra/ --name horse2zebra --model cycle_gan_plusplus
```
- The test results will be saved to a html file here: `./results/horse2zebra/latest_test/index.html`.

### Apply a pre-trained model (CycleGAN)
- You can download a pretrained model (e.g. horse2zebra) with the following script:
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. Check [here](https://github.com/junyanz/pytorch-CycleGANPlusPlus/blob/master/scripts/download_cyclegan_model.sh#L3) for all the available CycleGAN models.
- To test the model, you also need to download the  horse2zebra dataset:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Then generate the results using
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model cycle_gan_plusplus --no_dropout
```
- The option `--model cycle_gan_plusplus` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- For your own models, you may need to explicitly specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model.

## Citation
If you use this code for your research, please cite the original CycleGAN paper:
```
```
