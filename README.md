# Code for the Comenius University submission to the Track 1 of the 2021 AI City Challange

## Acknowledgments

This repository is based on the code for CenterTrack which can be found in the original repository: https://github.com/xingyizhou/CenterTrack

The DCNv2 code is adopted from https://github.com/MatthewHowe/DCNv2

This repository contains code and data for the 2D proximity and completeness fields from: https://github.com/Lijun-Yu/zero_virus

## Setup

For setup clone the repository and install the conda environment:

```shell script
git clone https://github.com/kocurvik/aicc
cd aicc
conda env create -f environment.yml
```

Activate the environment and then setup the DCNv2:
```shell script
conda activate aicc
cd model/networks/DCNv2
python setup.py build develop
```

The model should be downloaded from the CenterTrack [model zoo](https://github.com/xingyizhou/CenterTrack/blob/master/readme/MODEL_ZOO.md). Specifically [the MS COCO tracking model](https://drive.google.com/open?id=1tJCEJmdtYIh8VuN8CClGNws3YO7QGd40). Place the coco_tracking.pth file into the checkpoints directory.

## Running the code

To run on a single video use:

```shell script
python run_single_video_threaded.py /path/to/video.mp4
```

You can also try to use the serial version (slower):
```shell script
python run_single_video_threaded.py /path/to/video.mp4
```

To run the on the whole dataset in batched form run:
```shell script
python run_batch_threaded.py -b 2 /path/to/AIC21_Track1_Vehicle_Counting/Dataset_A
```

## Citation

If you find this code useful please consider citing our work:

```
@inproceedings{aicc2021comenius,
  title={Multi-Class Multi-Movement Vehicle Counting Based on CenterTrack},
  author={Kocur, Viktor and Ft\'{a}\v{c}nik, Milan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2021},
  month={June},
  pages = {4009-4015}
}
```
