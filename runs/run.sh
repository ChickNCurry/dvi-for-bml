python run.py --multirun \
common.h_dim=128 \
common.num_layers=6 \
common.non_linearity=SiLU,Tanh,GELU \
set_encoder.aggregation=max,mean \
set_encoder.use_context_size=False,True \
hydra/launcher=submitit_slurm \