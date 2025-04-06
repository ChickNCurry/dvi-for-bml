module load devel/miniconda
conda activate dvi-for-bml
which python

python ../run_dvinp.py --multirun \
hydra=gpu_dev \
model.model_variant=dis,dis_score,cmcd,ula \
model.context_variant=mean,bca,mhca \
model.contextual_schedules=true,false \
model.noise_variant=free,constr \
model.max_context_size=null,10 \
model.self_attn_num_heads=null,4 \