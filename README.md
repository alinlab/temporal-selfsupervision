# Time Is MattEr: Temporal Self-supervision for Video Transformers (TIME)

PyTorch implementation for <a href=https://icml.cc/virtual/2022/poster/18243>"Time Is MattEr: Temporal Self-supervision for Video Transformers"</a> (accepted in ICML 2022)

<p align="center">
<img width="950" alt="thumbnail" src="https://user-images.githubusercontent.com/34064646/178366422-f7db1073-81ef-46c3-889d-eb55046ef7f6.png">
</p>

## Requirements
First, create a conda virtual environment and activate it:
```
conda create -n TIME python=3.7 -y
source activate TIME
```
Then, install the following packages:
```
"torch>=1.9.0"
"torchvision>=0.10.0",
"einops>=0.3",
"yacs>=0.1.6",
"pyyaml>=5.1",
"imageio",
"fvcore",
"timm",
"scikit-learn",
"av",
"matplotlib",
"termcolor>=1.1",
"simplejson",
"tqdm",
"psutil",
"matplotlib",
"scikit-build",
"cmake",
"opencv-python",
"pandas",
"sklearn",
'torchmeta',
"ffmpeg-python",
"tensorboard",
```


### Training codes
# TimeSformer 
```
python main.py --cfg configs/timesformer/ssv2/train.yaml
```

# Motionformer
```
python main.py --cfg configs/motionformer/ssv2/train.yaml
```

# X-ViT
```
python main.py --cfg configs/xvit/ssv2/train.yaml
```

## Acknowledgement
Our code base is built partly upon the packages: 
<a href="https://github.com/facebookresearch/SlowFast">SlowFast</a>, <a href=https://github.com/facebookresearch/TimeSformer>TimeSformer</a>, and <a href=https://github.com/rwightman/pytorch-image-models>pytorch-image-models</a> by <a href=https://github.com/rwightman>Ross Wightman</a>.

## Citation
If you use this code for your research, please cite our papers.
```

```
