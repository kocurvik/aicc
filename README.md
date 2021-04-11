# The code for the Comenius University submission to the Track 1 of the 2021 AI City Challange

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

## Running the code

To run on a single video use:

```shell script
python run_single_video_threaded.py /path/to/video.mp4
```

To run the on the whole dataset in batched form run:
```shell script
python run_batch_threaded.py -b 2 /path/to/AIC21_Track1_Vehicle_Counting/Dataset_A
```
