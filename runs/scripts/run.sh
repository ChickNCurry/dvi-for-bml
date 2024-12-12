module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu \
wandb.project=cdvi-bml-alternating-new \
common.h_dim=64 \
set_encoder.aggregation=max,mean \
set_encoder.is_attentive=True,False \
training.max_clip_norm=null,0.5 \
training.alternating_ratio=null,0.1,0.2,0.3 \
training.alpha=null,1.0 \