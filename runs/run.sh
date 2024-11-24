python run.py --multirun \
hydra=gpu \
common.z_dim=2,4,6,8 \
common.h_dim=32,64,96,128 \
common.num_layers=6 \
common.non_linearity=SiLU \
set_encoder.aggregation=max,mean \
set_encoder.use_context_size=False \
set_encoder.is_attentive=True,False \
control_and_hyper_net.use_hyper_net=True,False \
control_and_hyper_net.is_cross_attentive=False \
training.max_clip_norm=0.2 \
wandb.project=dvi-for-bml-latent-space \
# training.num_epochs=1 \
# control_and_hyper_net.num_heads=1,2,3 \
