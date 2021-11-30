# Dynamic Slimmable Network (DS-Net)

This repository contains PyTorch code of our paper: [***Dynamic Slimmable Network***](https://arxiv.org/pdf/2103.13258.pdf) (CVPR 2021 Oral).

<p align="center">
<img width=95% alt="image" src="https://user-images.githubusercontent.com/61453811/113519958-59570000-95c2-11eb-8b0c-bb0f16ae3f89.png">
</p>
<p align="center">
Architecture of DS-Net. The width of each supernet stage is adjusted adaptively by the slimming ratio ρ predicted by the gate.
</p>
<p align="center">
<img width=55% alt="image" src="https://user-images.githubusercontent.com/61453811/113519893-cae27e80-95c1-11eb-9ec0-9d41ab54d25c.png">
</p>
<p align="center">
Accuracy vs. complexity on ImageNet.
</p>

## Pretrained Supernet
- [Supernet Checkpoint](https://github.com/changlin31/DS-Net/releases/download/v0.0.1/DS_MBNet-70_1.pth.tar)
- Here is a summary of sub-networks performance of the pretrained supernet:

  |    Subnetwork    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
  | ----------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  |       MAdds       | 133M  | 153M  | 175M  | 200M  | 226M  | 255M  | 286M  | 319M  | 355M  | 393M  | 433M  | 475M  | 519M  | 565M  |
  |      Top-1 (%)    | 70.1  | 70.4  | 70.8  | 71.2  | 71.6  | 72.0  | 72.4  | 72.7  | 73.0  | 73.3  | 73.6  | 73.9  | 74.1  | 74.6  |
  |      Top-5 (%)    | 89.4  |  89.6 | 89.9  | 90.2  | 90.3  | 90.6  | 90.9  | 91.0  | 91.2  | 91.4  | 91.5  | 91.7  | 91.8  | 92.0  |


## Usage
### 1. Requirements
- Install [PyTorch](http://pytorch.org/) 1.2.0+, for example:
  ```shell
  conda install -c pytorch pytorch torchvision
  ```
- Install [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) 0.3.2, for example:
  ```shell
  pip install timm==0.3.2
  ```
  
- Download ImageNet from http://image-net.org/. Move validation images to labeled subfolders using following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.shvalprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

### 2. Stage I: Supernet Training
 For example, train dynamic slimmable MobileNet supernet with 8 GPUs (takes about 2 days):
 ```
 python -m torch.distributed.launch --nproc_per_node=8 train.py /PATH/TO/ImageNet -c ./configs/mobilenetv1_bn_uniform.yml
 ```

### 3. Stage II: Gate Training
 - Modify `resume:` in `configs/mobilenetv1_bn_uniform_reset_bn.yml` to your supernet checkpoint. 
   Recalibrate BN before gate training
   ```
   python -m torch.distributed.launch --nproc_per_node=8 train.py /PATH/TO/ImageNet -c ./configs/mobilenetv1_bn_uniform_reset_bn.yml
   ```

 - Modify `resume:` in `configs/mobilenetv1_bn_uniform_gate.yml` to your supernet checkpoint after BN recalibration or our pretrained [Supernet Checkpoint](https://github.com/changlin31/DS-Net/releases/download/v0.0.1/DS_MBNet-70_1.pth.tar). 
   Start gate training
   ```
   python -m torch.distributed.launch --nproc_per_node=8 train.py /PATH/TO/ImageNet -c ./configs/mobilenetv1_bn_uniform_gate.yml
   ```

## Citation
If you use our code for your paper, please cite:
```bibtex
@inproceedings{li2021dynamic,
  author = {Changlin Li and
            Guangrun Wang and
            Bing Wang and
            Xiaodan Liang and
            Zhihui Li and
            Xiaojun Chang},
  title = {Dynamic Slimmable Network},
  booktitle = {CVPR},
  year = {2021}
}
```
