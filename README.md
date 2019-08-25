# ShelfNet-lightweight for paper (ShelfNet for fast semantic segmentation)
This repo contains implementation of ShelfNet-lightweight models for real-time models. <br/>
For ShelfNet implementation, refer to [another repo](https://github.com/juntang-zhuang/ShelfNet). <br/>

This repo is based on two implementations [Implementation 1](https://github.com/ycszen/TorchSeg) and [Implementation 2](https://github.com/CoinCheung/BiSeNet). This implementation takes about 24h on 2 GTX 1080Ti GPU. <br/>

## Link to results on Cityscapes test set
ShelfNet18-lw real-time: [https://www.cityscapes-dataset.com/anonymous-results/?id=b2cc8f49fc3267c73e6bb686425016cb152c8bc34fc09ac207c81749f329dc8d](https://www.cityscapes-dataset.com/anonymous-results/?id=b2cc8f49fc3267c73e6bb686425016cb152c8bc34fc09ac207c81749f329dc8d) <br/>
ShelfNet34-lw non real-time: [https://www.cityscapes-dataset.com/anonymous-results/?id=c0a7c8a4b64a880a715632c6a28b116d239096b63b5d14f5042c8b3280a7169d](https://www.cityscapes-dataset.com/anonymous-results/?id=c0a7c8a4b64a880a715632c6a28b116d239096b63b5d14f5042c8b3280a7169d)

## Data Preparation
Download fine labelled dataset from Cityscapes server, and decompress into ```./data``` folder. <br />
You might need to modify data path [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/train.py/#L58) and [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/evaluate.py/#L143)<br/>
```
$ mkdir -p data
$ mv /path/to/leftImg8bit_trainvaltest.zip data
$ mv /path/to/gtFine_trainvaltest.zip data
$ cd data
$ unzip leftImg8bit_trainvaltest.zip
$ unzip gtFine_trainvaltest.zip
```

## Two models and the pretrained weights
We provide two models, ShelfNet18 with 64 base channels for real-time semantic segmentation, and ShelfNet34 with 128 base channels for non-real-time semantic segmentation. <br/>Pretrained weights for [ShelfNet18](https://www.dropbox.com/s/84ol8lk99qcis9p/ShelfNet18_realtime.pth?dl=0) and [ShelfNet34](https://www.dropbox.com/s/q9jae02qe27wwa3/ShelfNet34_non_realtime.pth?dl=0).

## Requirements
PyTorch 1.0 or higher <br/>
python3 <br/>
scikit-image <br/>
tqdm<br/>

## How to run
Find the folder (```cd ShelfNet18_realtime``` or ```cd ShelfNet34_non_realtime```)

training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

evaluate on validation set (Create a folder called ```res```, this folder is automatically created if you train the model. Put checkpoint in ```res```folder, and make sure the checkpoint name and dataset path match ```evaluate.py```. Change checkpoint name to ```model_final.pth```by default)
```
python evaluate.py
```

## Running speed
test running speed of ShelfNet18-lw
```
python test_speed.py
```

You can modify the shape of input images to test running speed, by modifying [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/test_LWRF_speed.py#L32) <br />
You can test running speed of different models by modifying [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/test_LWRF_speed.py#L20) <br />
The running speed is an average of 100 single forward passes, therefore it's possible the speed varies. The code returns the mean running time by default.
