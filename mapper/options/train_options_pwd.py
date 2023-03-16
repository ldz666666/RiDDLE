from argparse import ArgumentParser
import os

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, default="experiments/exp_test",help='Path to experiment output directory')
		self.parser.add_argument('--use_dataset', default=False,action="store_true")
		self.parser.add_argument('--keep_optimizer', default=False,action="store_true")		
		self.parser.add_argument('--resume', default=False,action="store_true")		
		self.parser.add_argument('--save_training_data', default=True,action="store_true")	
		self.parser.add_argument('--latent_align_preprocess', default=False,action="store_true")
		self.parser.add_argument('--latent_avg_regularization', default=True,action="store_true")
		self.parser.add_argument('--latent_augment', default=False,action="store_true")			
		#self.parser.add_argument('--rev_dec_enc', default=False,action="store_true")		
		self.parser.add_argument('--lpips_type', default='vgg', type=str, help='LPIPS backbone')
		self.parser.add_argument('--mapper_type', default='simple', type=str, help='mapper type, [simple, transformer]')
		self.parser.add_argument('--use_landmark_prior', default=False,action="store_true")
		self.parser.add_argument('--transformer_normalize_type', type=str,default="layernorm")
		self.parser.add_argument('--transformer_split_list', nargs='+', type=int)
		self.parser.add_argument('--transformer_add_linear', default=False,action="store_true")		
		self.parser.add_argument('--transformer_add_pos_embedding', default=False,action="store_true")		

		self.parser.add_argument('--train_dataset_size', default=5000, type=int, help="Will be used only if no latents are given")
		self.parser.add_argument('--test_dataset_size', default=1000, type=int, help="Will be used only if no latents are given")

		self.parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--morefc_num', default=4, type=int, help='added fc num for simple mapper')

		self.parser.add_argument('--learning_rate', default=0.0005, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')

		self.parser.add_argument('--lpips_lambda', default=0, type=float, help='rev id factor')
		self.parser.add_argument('--landmark_lambda', default=0, type=float, help='landmark loss factor')
		
		self.parser.add_argument('--id_cos_margin', default=0.1, type=float, help='margin for id cosine similarity')
		self.parser.add_argument('--rev_id_enc_lambda', default=0, type=float, help='rev id factor')
		self.parser.add_argument('--rev_latent_enc_lambda', default=0, type=float, help='rev id factor')
		self.parser.add_argument('--latent_l2_enc_lambda', default=0, type=float, help='Latent L2 loss multiplier factor')
		
		self.parser.add_argument('--id_div_lambda', default=0, type=float, help='rev id factor')
		self.parser.add_argument('--rev_id_dec_lambda', default=0, type=float, help='rev id factor')
		self.parser.add_argument('--rev_id_intra_lambda', default=1, type=float, help='rev id factor')
		self.parser.add_argument('--rev_latent_dec_lambda', default=0, type=float, help='rev id factor')
		self.parser.add_argument('--latent_l2_dec_lambda', default=0, type=float, help='Latent L2 loss multiplier factor')
		self.parser.add_argument('--recon_id_lambda', default=0, type=float, help='id perceptual loss factor')		
		self.parser.add_argument('--recon_pix_lambda', default=0, type=float, help='lpips factor')
		self.parser.add_argument('--recon_latent_lambda', default=0, type=float, help='pixelwise l2 loss factor')

		self.parser.add_argument('--latent_align_lambda', default=0, type=float, help='Dw loss multiplier')
		self.parser.add_argument('--latent_avg_lambda', default=0, type=float, help='Dw loss multiplier')
		self.parser.add_argument('--parse_lambda', default=0, type=float, help='parse loss multiplier for mouth and eye')

		self.parser.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
		self.parser.add_argument('--w_discriminator_lr', default=2e-5, type=float, help='Dw learning rate')
		self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
		self.parser.add_argument("--d_reg_every", type=int, default=16,
								help="interval for applying r1 regularization")
		self.parser.add_argument('--use_w_pool', action='store_true',
								help='Whether to store a latnet codes pool for the discriminator\'s training')
		self.parser.add_argument("--w_pool_size", type=int, default=50,
								help="W\'s pool size, depends on --use_w_pool")

		self.parser.add_argument('--latents_train_path', default="embeddings/invert_w_256.pt", type=str, help="The latents for the training")
		self.parser.add_argument('--latents_test_path', default="embeddings/invert_w_256.pt", type=str, help="The latents for the validation")
		self.parser.add_argument('--id_emb_train_path', default="embeddings/id_embedding_256.pt", type=str, help="The id embedding for the training")
		self.parser.add_argument('--id_emb_test_path', default="embeddings/id_embedding_256.pt", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--encoder_type', default="Encoder4Editing", type=str, help="e4e model path")

		self.parser.add_argument('--psp_weight_path', default="pretrained_models/e4e_ffhq_encode_256.pt", type=str, help="The latents for the validation")
		self.parser.add_argument('--parsenet_weights', default='pretrained_models/parsenet.pth', type=str, help='Path to Parsing model weights')
		self.parser.add_argument('--landmark_encoder_weights', default='pretrained_models/mobilefacenet_model_best.pth.tar', type=str, help='Path to landmark encoder weights')
		self.parser.add_argument('--stylegan_weights', default='pretrained_models/stylegan2-ffhq-256.pt', type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_size', default=256, type=int)
		self.parser.add_argument('--ir_se50_weights', default='pretrained_models/model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")
		self.parser.add_argument('--resume_checkpoint_path', default=None, type=str, help='checkpoint path')

		self.parser.add_argument('--max_steps', default=100000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=10, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=20, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')

		#distribution related
		self.parser.add_argument('--ddp', default=False,action="store_true")
		self.parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

		'''
		discarded
		self.parser.add_argument('--multi_level_feature_injection', default=False,action="store_true")
		self.parser.add_argument('--clamp_latent_delta', default=False, action="store_true")
		self.parser.add_argument('--infogan', default=False,action="store_true")
		'''

	def parse(self):
		opts = self.parser.parse_args()
		return opts