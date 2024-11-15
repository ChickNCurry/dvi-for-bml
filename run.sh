python run.py --multirun \
common.h_dim=128 \
common.num_layers=6 \
common.non_linearity=SiLU,Tanh \
control_function.is_cross_attentive=False,True \
set_encoder.aggregation=max,mean \
hydra/launcher=submitit_slurm \