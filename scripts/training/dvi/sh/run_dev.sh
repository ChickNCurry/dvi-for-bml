source /home/ka/ka_anthropomatik/ka_km6619/dvi-for-bml/.venv/bin/activate
which python

python ../run_dvi.py --multirun \
hydra=gpu_dev \
model.model_variant=dis \
model.context_variant=mean \
model.contextual_schedules=true \
model.noise_variant=free \
model.max_context_size=10 \
model.self_attn_num_heads=4 \