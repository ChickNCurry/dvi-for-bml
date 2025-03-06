module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_dvinp.py --multirun \
hydra=gpu \
wandb.project=cluster-dvinp-linesine \
model.context_variant=mean,bca \
model.noise_variant=free,cos \
model.contextual_schedules=true,false \
model.model_variant=dis,dis_score \
model.num_steps=16,32 \
training.trainer_variant=cntxt,fwdcntxt \
# model.self_attn_num_heads=null,1 \