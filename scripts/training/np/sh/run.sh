source /home/ka/ka_anthropomatik/ka_km6619/dvi-for-bml/.venv/bin/activate
which python

python ../run_np.py --multirun \
hydra=gpu \
model.model_variant=cnp,lnp \
model.context_variant=mean,max,bca \
model.self_attn_num_heads=null,8 \
training.trainer_variant=data,target,context \
# training.seed=0,1 \