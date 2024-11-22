python run.py --multirun \
hydra=gpu \
common.h_dim=128,64 \
common.num_layers=6 \
hyper_net.use_hyper_net=True,False \
common.non_linearity=SiLU,Tanh,GELU \
set_encoder.aggregation=max,mean \
set_encoder.use_context_size=False,True \
training.max_clip_norm=0.1,0.5 \
# wandb.project=test \
# training.num_epochs=1 \
