python run.py --multirun \
hydra=gpu_dev \
common.z_dim=2 \
common.h_dim=64,128 \
common.num_layers=6 \
common.non_linearity=SiLU \
set_encoder.aggregation=max,mean \
set_encoder.use_context_size=False \
set_encoder.is_attentive=True,False \
control_and_hyper_net.use_hyper_net=False \
control_and_hyper_net.is_cross_attentive=False \
training.max_clip_norm=None,0.5 \
training.alternating_ratio=None,0.3,0.5,0.7 \
wandb.project=test \
training.num_epochs=1 \
