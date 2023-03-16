
cd /hd3/lidongze/animation/RiDDLE


exp_dir=/hd3/lidongze/animation/RiDDLE/experiments/exp_test


CUDA_VISIBLE_DEVICES=$1 \
python mapper/training/coach_id_swapping_password_same_mapper.py \
--exp_dir $exp_dir \
--keep_optimizer \
--lpips_lambda 1 \
--landmark_lambda 0 \
--use_dataset \
--id_cos_margin 0.1 \
--rev_id_enc_lambda 1 \
--rev_latent_enc_lambda 0 \
--rev_id_intra_lambda 0 \
--latent_l2_enc_lambda 0 \
--rev_id_dec_lambda 1 \
--rev_latent_dec_lambda 0 \
--latent_l2_dec_lambda 0 \
--recon_id_lambda 1 \
--recon_pix_lambda 0.05 \
--recon_latent_lambda 0 \
--image_interval 500 \
--latent_align_lambda 0 \
--parse_lambda 0 \
--mapper_type transformer \
--transformer_normalize_type layernorm \
--transformer_split_list 4 4 6 \
--w_discriminator_lambda 0 \
--latent_l2_dec_lambda 0.0 \
--latent_l2_enc_lambda 0.0 \
--latent_avg_lambda 0 \
--latents_train_path /hd3/lidongze/data/ffhq/invert_w_256.pt \
--morefc_num 0 \
--batch_size 4