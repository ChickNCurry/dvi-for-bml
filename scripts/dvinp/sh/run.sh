module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_dvinp.py --multirun \
hydra=gpu \
wandb.project=cluster-dvinp-noscore \
training.seed=0 \
training.max_clip_norm=1.0 \
model.self_attn_num_heads=null,1 \
model.context_variant=mean,bca \
model.noise_variant=free,cos \
model.contextual_schedules=true,false \
training.trainer_variant=context,forward,forwardandcontext \