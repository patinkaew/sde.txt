python diffusion_clip.py \
--text "blue car" \
--image cifar10-images/4.png \
--t0 1000 \
--s_inv 1500 \
--s_gen 60 \
--nudge_iter 1000 \
--id_weight 1. \
--save_path result/diffclip3 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--log_every 50
