#!/bin/sh
sbatch --output="/data/raear/logs/%j.out" --error="/data/raear/logs/%j.out" --partition=gpu --cpus-per-task=8 --gres=gpu:p100:1 --mem=64g --time=10-00:00:00 train.sh