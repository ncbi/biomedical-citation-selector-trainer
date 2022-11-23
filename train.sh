#!/bin/sh
eval "$(/data/raear/miniconda3/bin/conda shell.bash hook)"
conda activate retraining
python -m BmCS.retrain --workdir /data/raear/working_dir/bmcs-trainer-v2