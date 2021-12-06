python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "truck" \
--constrast_texts "airplane, automobile, bird, cat, deer, dog, frog, horse, ship" \
--cond_scaling 1000 \
--guiding_start 0 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_0_start_0_scale_1000
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "truck" \
--constrast_texts "airplane, automobile, bird, cat, deer, dog, frog, horse, ship" \
--cond_scaling 1000 \
--guiding_start 1000 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_0_start_1000_scale_1000
