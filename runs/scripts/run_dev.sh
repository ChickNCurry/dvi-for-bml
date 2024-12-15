module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu_dev \
wandb.project=test \
common.h_dim=64 \
set_encoder.aggregation=max \
set_encoder.is_attentive=True \
training.max_clip_norm=null \
# training.num_epochs=1 \
