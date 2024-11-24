python run.py --multirun \
hydra=gpu \
common.z_dim=2,4,8,16 \
common.h_dim=64 \
common.num_layers=6 \
common.non_linearity=SiLU \
set_encoder.aggregation=mean \
set_encoder.use_context_size=False \
control_and_hyper_net.use_hyper_net=True \
control_and_hyper_net.is_cross_attentive=True,False \
control_and_hyper_net.num_heads=1,2,3 \
training.max_clip_norm=0.5 \
wandb.project=dvi-for-bml-cross-attn \
# training.num_epochs=1 \
