.venv/bin/python scripts/run.py --multirun \
common.h_dim=128 \
common.num_layers=6 \
common.non_linearity=SiLU,Tanh \
dvi_process.is_cross_attentive=False,True \
set_encoder.aggregation=max,mean \
wandb_logging=False