python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "airplane" \
--constrast_texts "automobile, bird, cat, deer, dog, frog, horse, ship, truck" \
--cond_scaling 1 \
--guiding_start 1000 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_1_start_1000_scale_1
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "airplane" \
--constrast_texts "automobile, bird, cat, deer, dog, frog, horse, ship, truck" \
--cond_scaling 10 \
--guiding_start 1000 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_1_start_1000_scale_10
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "airplane" \
--constrast_texts "automobile, bird, cat, deer, dog, frog, horse, ship, truck" \
--cond_scaling 100 \
--guiding_start 1000 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_1_start_1000_scale_100
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "airplane" \
--constrast_texts "automobile, bird, cat, deer, dog, frog, horse, ship, truck" \
--cond_scaling 250 \
--guiding_start 1000 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_1_start_1000_scale_250
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "airplane" \
--constrast_texts "automobile, bird, cat, deer, dog, frog, horse, ship, truck" \
--cond_scaling 500 \
--guiding_start 1000 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_1_start_1000_scale_500
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "airplane" \
--constrast_texts "automobile, bird, cat, deer, dog, frog, horse, ship, truck" \
--cond_scaling 1000 \
--guiding_start 1000 \
--config config_yml/cifar10.yml \
--ckpt model_ckpt/cifar10.ckpt \
--save_path constrast_result/cifar10_constrast_1_start_1000_scale_1000
