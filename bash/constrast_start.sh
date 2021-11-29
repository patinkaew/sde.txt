python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "old woman" \
--constrast_texts "young man, young boy, young girl, old man" \
--cond_scaling 1 \
--guiding_start 1000 \
--save_path constrast_result/constrast_3_start_0
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "old woman" \
--constrast_texts "young man, young boy" \
--cond_scaling 1 \
--guiding_start 1000 \
--save_path constrast_result/constrast_2_start_0
python constrast_cond_gen.py \
--seed 236 \
--use_seed 1 \
--target_text "old woman" \
--constrast_texts "young man" \
--cond_scaling 1 \
--guiding_start 1000 \
--save_path constrast_result/constrast_1_start_0
