from argparse import ArgumentParser
import os

class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--e4e_model_weights', default= "pretrained_models/e4e_ffhq_encode_256.pt", type=str, help="The latents for the training")
		self.parser.add_argument('--stylegan_weights', default= "pretrained_models/stylegan2-ffhq-256.pt", type=str, help="The latents for the training")
		self.parser.add_argument('--stylegan_size', default= 256, type=int, help="stylegan image size")
		self.parser.add_argument('--ir_se50_weights', default='pretrained_models/model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")
		self.parser.add_argument('--id_cos_margin', default=0.1, type=float, help='margin for id cosine similarity')
		self.parser.add_argument('--mapper_weight', default="pretrained_models/iteration_90000.pt",type=str, help="The latents for the validation")
		self.parser.add_argument('--mapper_type', default='transformer', type=str, help="The latents for the validation")
		self.parser.add_argument('--transformer_normalize_type', default="layernorm", type=str, help="The latents for the validation")
		self.parser.add_argument('--transformer_split_list', nargs='+', type=int,default=[4,4,6])
		self.parser.add_argument('--transformer_add_linear', default=True,action="store_false")		
		self.parser.add_argument('--transformer_add_pos_embedding', default=True,action="store_false")		
		self.parser.add_argument('--latent_path', default="embeddings/invert_w_256.pt", type=str, help="The id embedding for the training")
		self.parser.add_argument('--image_path', default="/hd3/lidongze/data/ffhq/ffhqfirst10_images", type=str, help="The id embedding for the training")
		self.parser.add_argument('--exp_dir', default="experiments/exp_test", type=str, help="The id embedding for the testing")
		self.parser.add_argument('--image_size', default=256, type=int, help="The latents for the validation")
		self.parser.add_argument('--pwd_num', default=6, type=int, help="The latents for the validation")
		self.parser.add_argument('--batch_size', default=4, type=int, help="The latents for the validation")
		self.parser.add_argument('--latents_num', default=14, type=int, help="The latents for the validation")
		self.parser.add_argument('--morefc_num', default=0, type=int, help="The latents for the validation")
		self.parser.add_argument('--interpolate_batch_num', default=-1, type=int, help="The latents for the validation")
		self.parser.add_argument('--device', default="cuda", type=str, help="The id embedding for the testing")


	def parse(self):
		opts = self.parser.parse_args()
		return opts