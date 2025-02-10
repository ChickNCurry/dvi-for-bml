module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run.py --multirun \
hydra=gpu_dev \
wandb.project=cluster-noscore-test \
training.seed=1,2,3 \
training.max_clip_norm=null,1.0,0.1 \
model.self_attn_num_heads=null,1 \
model.model_variant=bca,aggr \
model.noise_variant=free,cos \
model.contextual_schedules=true,false \
# training.alpha=null,1.0 \