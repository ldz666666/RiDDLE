
cd /hd3/lidongze/animation/RiDDLE


exp_dir=/hd3/lidongze/animation/RiDDLE/experiments/exp_test_ddp

# python -m torch.distributed.launch \
# --nnodes 1 \
# --nproc_per_node 4 \
# --node_rank 0 \
# --master_port 6005 \

CUDA_VISIBLE_DEVICES=8,9 \
python -m torch.distributed.launch \
--nnodes 1 \
--nproc_per_node 2 \
--node_rank 0 \
--master_port 6005 \
mapper/training/coach_id_swapping_password_same_mapper.py \
--exp_dir $exp_dir \
--keep_optimizer \
--lpips_lambda 1 \
--landmark_lambda 0.0 \
--id_cos_margin 0.1 \
--rev_id_enc_lambda 1 \
--rev_latent_enc_lambda 0 \
--latent_l2_enc_lambda 0 \
--rev_id_dec_lambda 1 \
--rev_latent_dec_lambda 0 \
--latent_l2_dec_lambda 0 \
--recon_id_lambda 1 \
--recon_pix_lambda 0.05 \
--recon_latent_lambda 0 \
--image_interval 500 \
--latent_align_lambda 0 \
--parse_lambda 0.1 \
--mapper_type simple \
--w_discriminator_lambda 0 \
--latent_l2_dec_lambda 0 \
--latent_l2_enc_lambda 0 \
--mapper_type simple \
--morefc_num 0 \
--batch_size 2