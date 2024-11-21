python run.py --multirun \
hydra/launcher=submitit_slurm \
common.h_dim=128 \
common.num_layers=6 \
hyper_net.use_hyper_net=True,False \
common.non_linearity=SiLU,Tanh,GELU \
set_encoder.aggregation=max,mean \
set_encoder.use_context_size=False,True \
