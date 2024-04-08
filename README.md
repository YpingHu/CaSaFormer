# CaSaFormer
This is the code for our paper "CaSaFormer: A Cross- and Self-attention Based Lightweight Network for Large-scale Building Semantic Segmentation"
# Install
MMCV >=1.3.13 and <=1.5.0
`pip install -r requirements.txt`
# Training
`python /tools/train.py --work-dir casaformer --gpus 1 --seed 26 ..\configs\casaformer.py` 
# Testing
`python /tools/test.py --work-dir casaformer  --eval mIoU mFscore --gpu-collect ..\configs\casaformer.py casaformer\iter_xxx.pth`
