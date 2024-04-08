# CaSaFormer
This is the code for our paper "CaSaFormer: A Cross- and Self-attention Based Lightweight Network for Large-scale Building Semantic Segmentation"
# Training
`python /tools/train.py --work-dir casaformer --gpus 1 --seed 26 ..\configs\casaformer.py` 
# Testing
`python /tools/test.py --work-dir casaformer  --eval mIoU mFscore --gpu-collect ..\configs\casaformer.py ..\configs\casaformer.py casaformer\iter_xxx.pth`
