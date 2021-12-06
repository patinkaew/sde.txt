# +
python compute_KID_score_cifar10.py \
--mode "eval" \
--num_images 500 \
--generate_img_dir "cifar10_images/airplane" \
--real_img_dir "cifar10_gen_exp" \
--subset_size 10

python compute_KID_score_cifar10.py \
--mode "eval" \
--num_images 500 \
--generate_img_dir "cifar10_images/airplane" \
--real_img_dir "cifar10_gen_base" \
--subset_size 10

python compute_KID_score_cifar10.py \
--mode "eval" \
--num_images 500 \
--generate_img_dir "cifar10_images/airplane" \
--real_img_dir "cifar10_gen_null" \
--subset_size 10
