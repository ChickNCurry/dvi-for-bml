module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_dvinp.py --multirun \
hydra=gpu \
wandb.project=cluster-score \
training.seed=1,2 \
training.max_clip_norm=1.0,0.1 \
model.self_attn_num_heads=null,1 \
model.model_variant=bca,mean \
model.noise_variant=free,cos \
model.contextual_schedules=true,false \
# training.alpha=null,1.0 \