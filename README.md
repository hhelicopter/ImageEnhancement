# Low-illumination image enhancement using a conditional generative adversarial network
This repository is a GAN model for Low-illumination image ehancement
## Dependence
- tensorflow==1.8.0
- numpy==1.16.0
- pillow==6.2.2
## How to use
1. Create ./log and ./result at first.
2. The training iamges saved in ./dataset/high and ./dataset/low.The testing iamges saved in ./dataset/test. 
3. 'python3 train.py' for training.
4. You can modify the parameters in train.py before training.
5. 'python3 test.py' for testing.
6. The weight of GAN is saved in ./log and the result of testing is saved in ./result.
