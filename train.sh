eval "$(/data/raear/miniconda3/bin/conda shell.bash hook)"
conda activate retraining
sbatch --cpus-per-task=8 --gres=gpu:p100:1 --mem=64g --time=10:00:00:00 train.sh