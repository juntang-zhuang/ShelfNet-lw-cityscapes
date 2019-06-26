# anonymous-repo
Anonymous repo to reproduce fast semantic segmentation results.
This repo is based on two implementations of [BiSeNet1](https://github.com/CoinCheung/BiSeNet) and [TorchSeg](https://github.com/ycszen/TorchSeg), but we are not affiliated with any of them.

## Data Preparation
Download fine labelled dataset from Cityscapes server, and decompress into ```./data``` folder

## Two models and the pretrained weights
We provide two models, ShelfNet18 with 32 base channels for real-time semantic segmentation, and ShelfNet34 with 64 base channels for non-real-time semantic segmentation. The pre-trained weights are available:<br />
[ShelfNet18](https://www.dropbox.com/s/ozfaxuo8be610ko/ShelfNet18_realtime.pth?dl=0) <br />
[ShelfNet34](https://www.dropbox.com/s/az41ud5qipp173c/ShelfNet34_non_realtime.pth?dl=0) <br />

## How to run with PyTorch 1
training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

evalate on validation set
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
You can test running speed of different models by modifying [here](https://github.com/NoName-sketch/anonymous/blob/master/ShelfNet18_realtime/test_LWRF_speed.py#L20)
