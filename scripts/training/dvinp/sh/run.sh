source /home/ka/ka_anthropomatik/ka_km6619/dvi-for-bml/.venv/bin/activate
which python

python ../run_dvinp.py --multirun \
hydra=gpu \
training.trainer_variant=cntxt,fwdcntxt \
model.noise_variant=free,constr \
model.context_variant=mean,bca,max \
model.model_variant=dis,dis_score \
training.seed=0,1 \