# RiDDLE (CVPR 2023)


<img width="500" alt="RiDDLE" src="https://user-images.githubusercontent.com/41360226/225599374-39e81c5e-01b1-462a-82ab-0c5bde0152bb.png">


### [Arxiv](https://arxiv.org/abs/2303.05171)

Author implementation of RiDDLE: Reversible and Diversified De-identification with Latent Encryptor 

[Dongze Li](https://ldz666666.github.io/), Wei Wang, Kang Zhao, Jing Dong and Tieniu Tan 

## Abstract 
 This work presents RiDDLE, short for Reversible and Diversified De-identification with Latent Encryptor, to protect the identity information of people from being misused. Built upon a pre-learned StyleGAN2 generator, RiDDLE manages to encrypt and decrypt the facial identity within the latent space. The design of RiDDLE has three appealing properties. First, the encryption process is cipher-guided and hence allows diverse anonymization using different passwords. Second, the true identity can only be decrypted with the correct password, otherwise the system will produce another de-identified face to maintain the privacy. Third, both encryption and decryption share an efficient implementation, benefiting from a carefully tailored lightweight encryptor. Comparisons with existing alternatives confirm that our approach accomplishes the de-identification task with better quality, higher diversity, and stronger reversibility. We further demonstrate the effectiveness of RiDDLE in anonymizing videos. Code and models will be made publicly available.

## Pipeline
<img width="700" alt="RiDDLE_pipeline" src="https://user-images.githubusercontent.com/41360226/225602427-3a17a937-bce8-4b32-9f75-7151a0fd2966.png">

## Usage 

### Environment
To run the inference and the training scripts, first you need to set up a virtual environment by `conda env create -f RiDDLE.yaml`

### Inference
To perform identity encryption and decryption, just `python coach_test.py`

### Train
To train a latent encryptor, just `sh scripts/run_coach_id_pwd_same.sh`

We also support Distributed Data Parallel (DDP) training, `sh scripts/run_coach_id_pwd_same_ddp.sh`. 

### Data and Pretrain models
Our data and pretrain models can be found at [this link](https://pan.baidu.com/s/1Yf65Q8wah3N305MttL2B8g), password is `sqp8`


## Video De-identification Results

We combine our method with [Stitch Tuning](https://github.com/rotemtzaban/STIT) to de-identify videos. 

### Curry


https://user-images.githubusercontent.com/41360226/222632925-0c373a53-0309-4c8b-9deb-b3766702d65d.mov


### Lebron


https://user-images.githubusercontent.com/41360226/222632879-b2525e5f-d60e-4438-98c9-b9b6d71559f9.mov

### Jim


https://user-images.githubusercontent.com/41360226/222632899-9941b004-6078-461c-b987-ccef903f4c5a.mov



## Citation
If you found this repo useful, please cite
```
@inproceedings{li2023riddle,
  title={RiDDLE: Reversible and Diversified De-identification with Latent Encryptor},
  author={Li, Dongze and Wang, Wei and Zhao, Kang and Dong, Jing and Tan, Tieniu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2023}
}
``` 

## Acknowledgements
Some parts of our code is based on [HairCLIP](https://github.com/wty-ustc/HairCLIP) and [e4e](https://github.com/omertov/encoder4editing), thanks for their great work.
