# Introduction

This is the implementation of 3D CNN pruning in the paper "[Hardware-Friendly 3D CNN Acceleration With Balanced Kernel Group Sparsity](https://ieeexplore.ieee.org/document/10502121)" accepted by IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2024.

- Pruning sparisity types include "kgr" (kernel group row), "kgc" (kernel group column), and both of them (KGRC) in weight kernel groups.
- Supported networks include C3D and R(2+1)D (from torchvision). The dataset for evaluation is UCF101.


# Brief Instructions

- `python>=3.6, <=3.7`
- Install dependencies: `pip install -r requirements.txt`
    - `torch>=1.7, <=1.10.1`
- Download the dataset. Converting into LMDB files is recommended.
- Baseline models should be in `checkpoints/baseline` directory, and pruned models will be saved into `checkpoints` directory.
- Perform pruning by running scripts in `scripts` directory: e.g., `bash scripts/kgrc_c3d.sh`


# Datasets

Download and process the dataset with the following commands in `datasets` directory.
<!-- - (only for reference) https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md -->

- Download UCF101 original video dataset from https://www.crcv.ucf.edu/data/UCF101/UCF101.rar and unpack to `ucf101` directory. Download the train/test splits for action recognition from https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip and unpack.
  - To unpack `rar` files, install `unrar` by running `sudo apt install unrar`.
```sh
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
mkdir ucf101 && unrar e UCF101.rar ucf101
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zi --no-check-certificate && unzip UCF101TrainTestSplits-RecognitionTask.zip
```
- Extract frames from videos by running `python extract_videos.py -d ucf101`.
- Convert frames into LMDB files with `create_lmdb.py`.
```sh
python create_lmdb.py -d ucf101_frame -s train -vr 0 9437
python create_lmdb.py -d ucf101_frame -s val -vr 0 3755
```
```sh
datasets/ucf101_frame/
├── v_ApplyEyeMakeup_g01_c01
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── 00004.jpg
│   ├── 00005.jpg
```


# Baseline Models

Download the unpruned baseline models and put them in `checkpoints/baseline` directory.
- C3D: [Baidu Netdisk](https://pan.baidu.com/s/15-8MFTi-chQGglprKX6SnA) (Password: loj2)
- R(2+1)D: [Baidu Netdisk](https://pan.baidu.com/s/1JuTTCL5Jq3GrBEZjugrUig) (Password: zmh0)
