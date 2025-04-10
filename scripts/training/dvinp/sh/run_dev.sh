source /home/ka/ka_anthropomatik/ka_km6619/dvi-for-bml/.venv/bin/activate
which python

python ../run_dvinp.py --multirun \
hydra=cpu \
model.model_variant=dis \
model.context_variant=mean \
model.noise_variant=free \
training.trainer_variant=cntxt \
