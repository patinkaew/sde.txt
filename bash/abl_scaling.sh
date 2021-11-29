python ablation_cond_gen.py \
--text tanned \
--cond_scaling 10 \
--save_path ablation_result/scale_10

python ablation_cond_gen.py \
--text tanned \
--cond_scaling 100 \
--save_path ablation_result/scale_100

python ablation_cond_gen.py \
--text tanned \
--cond_scaling 1000 \
--save_path ablation_result/scale_1000
