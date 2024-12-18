module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu_dev \
wandb.project=test \
set_encoder.aggregation=max \
set_encoder.is_attentive=True \
training.max_clip_norm=null \
control.is_cross_attentive=True \
training.alpha=1.0 \