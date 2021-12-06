# +
python collect_final_images.py \
--src_dir "constrast_result/cifar10_contrast_final" \
--dst_dir "cifar10_gen_exp"

python collect_final_images.py \
--src_dir "constrast_result/cifar10_spherical_final" \
--dst_dir "cifar10_gen_base"

python collect_final_images.py \
--src_dir "constrast_result/cifar10_null_final" \
--dst_dir "cifar10_gen_null"
