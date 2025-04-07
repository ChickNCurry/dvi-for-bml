module load devel/miniconda
conda activate dvi-for-bml
which python
# source venv/bin/activate
# which python

python ../run_dvi.py --multirun \
hydra=gpu_dev \
model.model_variant=dis \
model.context_variant=mean \
model.contextual_schedules=true \
model.noise_variant=free \
model.max_context_size=10 \
model.self_attn_num_heads=4 \