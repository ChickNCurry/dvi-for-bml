source /home/ka/ka_anthropomatik/ka_km6619/dvi-for-bml/.venv/bin/activate
which python

python ../run_dvinp.py --multirun \
hydra=gpu \
model.model_variant=dis,dis_score \
model.context_variant=mean,bca \
model.noise_variant=free,constr \
training.trainer_variant=cntxt,fwdcntxt \