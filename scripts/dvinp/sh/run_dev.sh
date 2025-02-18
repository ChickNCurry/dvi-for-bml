module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_dvinp.py --multirun \
hydra=gpu_dev \
wandb.project=cluster-noscore-test \
training.seed=1 \
training.max_clip_norm=null \
model.self_attn_num_heads=null \
model.model_variant=bca \
model.noise_variant=free \
model.contextual_schedules=true \
training.num_epochs=1 \
# training.alpha=null,1.0 \