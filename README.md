# Anonymous-repo
Anonymous repo to reproduce fast semantic segmentation results.
This repo is based on two implementations of [BiSeNet1](https://github.com/CoinCheung/BiSeNet) and [TorchSeg](https://github.com/ycszen/TorchSeg), but we are not affiliated with any of them.

## Data Preparation
Download fine labelled dataset from Cityscapes server, and decompress into ```./data``` folder. <br />
You might need to midify data path [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/train.py/#L58) and [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/evaluate.py/#L143)

## Two models and the pretrained weights
We provide two models, ShelfNet18 with 32 base channels for real-time semantic segmentation, and ShelfNet34 with 64 base channels for non-real-time semantic segmentation. The pre-trained weights are available:<br />
[Res18 backbone](https://www.dropbox.com/s/ozfaxuo8be610ko/ShelfNet18_realtime.pth?dl=0) <br />
[Res34 backbone](https://www.dropbox.com/s/az41ud5qipp173c/ShelfNet34_non_realtime.pth?dl=0) <br />
You might need to change checkpoint names to match the file name in ```evaluate.py```

## How to run with PyTorch 1
training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

evaluate on validation set
```
python evaluate.py
```

## Running speed
test running speed of ShelfNet
```
python test_speed.py
```

test running speed of Lightweight refinenet
```
python test_LWRF_speed.py
```

You can modify the shape of input images to test running speed, by modifying [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/test_LWRF_speed.py#L32) <br />
You can test running speed of different models by modifying [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/test_LWRF_speed.py#L20) <br />
The running speed is an average of 100 single forward passes, therefore it's possible the speed varies. The code returns the mean running time by default.
