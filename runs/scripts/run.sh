module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu \
wandb.project=dvi-np-cluster-informed-cross-attn \
set_encoder.aggregation=max,mean \
set_encoder.is_attentive=False,True \
control.is_cross_attentive=False,True \
training.max_clip_norm=null,0.2 \
training.alpha=null,1.0 \