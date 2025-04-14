source /home/ka/ka_anthropomatik/ka_km6619/dvi-for-bml/.venv/bin/activate
which python

python ../run_np.py --multirun \
hydra=cpu \
model.model_variant=lnp \
model.context_variant=mean \
model.self_attn_num_heads=8 \
training.trainer_variant=data \
training.seed=0 \