# RiDDLE (CVPR 2023)


<img width="600" alt="RiDDLE" src="https://user-images.githubusercontent.com/41360226/225599374-39e81c5e-01b1-462a-82ab-0c5bde0152bb.png">

### [Arxiv](https://arxiv.org/abs/2303.05171)

Author implementation of RiDDLE: Reversible and Diversified De-identification with Latent Encryptor 

[Dongze Li](https://ldz666666.github.io/), Wei Wang, Kang Zhao, Jing Dong and Tieniu Tan 

## Abstract 
 This work presents RiDDLE, short for Reversible and Diversified De-identification with Latent Encryptor, to protect the identity information of people from being misused. Built upon a pre-learned StyleGAN2 generator, RiDDLE manages to encrypt and decrypt the facial identity within the latent space. The design of RiDDLE has three appealing properties. First, the encryption process is cipher-guided and hence allows diverse anonymization using different passwords. Second, the true identity can only be decrypted with the correct password, otherwise the system will produce another de-identified face to maintain the privacy. Third, both encryption and decryption share an efficient implementation, benefiting from a carefully tailored lightweight encryptor. Comparisons with existing alternatives confirm that our approach accomplishes the de-identification task with better quality, higher diversity, and stronger reversibility. We further demonstrate the effectiveness of RiDDLE in anonymizing videos. Code and models will be made publicly available.


## Usage 

### Environment
To set up a virtual environment, just `conda env create -f RiDDLE.yaml`

### Inference
To perform identity encryption and decryption, just `python coach_test.py`

### Train
To train a latent encryptor, just `sh scripts/run_coach_id_pwd_same.sh`

### Data and Pretrain models
Our data and pretrain models can be found at [this link](https://pan.baidu.com/s/1Yf65Q8wah3N305MttL2B8g), password is sqp8


 
## Video De-identification Results

### Curry


https://user-images.githubusercontent.com/41360226/222632925-0c373a53-0309-4c8b-9deb-b3766702d65d.mov


### Lebron


https://user-images.githubusercontent.com/41360226/222632879-b2525e5f-d60e-4438-98c9-b9b6d71559f9.mov

### Jim


https://user-images.githubusercontent.com/41360226/222632899-9941b004-6078-461c-b987-ccef903f4c5a.mov



## Citation
If you found this repo useful, please cite
```
@article{li2023riddle,
  title={RiDDLE: Reversible and Diversified De-identification with Latent Encryptor},
  author={Li, Dongze and Wang, Wei and Zhao, Kang and Dong, Jing and Tan, Tieniu},
  journal={arXiv preprint arXiv:2303.05171},
  year={2023}
}
``` 
