module load devel/miniconda
conda activate dvi-for-bml
which python
# source venv/bin/activate
# which python

python ../run_dvi.py --multirun \
hydra=cpu \
model.model_variant=dis,dis_score \
model.context_variant=mean,bca,mhca \
model.contextual_schedules=true,false \
model.noise_variant=free,constr \
model.max_context_size=null,10 \
model.self_attn_num_heads=null,4 \