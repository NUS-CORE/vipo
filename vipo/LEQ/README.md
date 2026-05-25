# Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning

This repository contains the official implementation of [Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning](https://kwanyoungpark.github.io/LEQ/) by [Kwanyoung Park](https://kwanyoungpark.github.io/) and [Youngwoon Lee](https://youngwoon.github.io/).

If you use this code for your research, please consider citing our paper:
```
@article{park2024tackling,
  title={Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning},
  author={Kwanyoung Park and Youngwoon Lee},
  journal={arXiv Preprint arxiv:2407.00699},
  year={2024}
}
```

## How to run the code

### Install dependencies

```bash
conda create -n LEQ python=3.9
conda activate LEQ

pip install -r requirements.txt

# Install jax (https://github.com/google/jax#pip-installation-gpu-cuda)
pip install jax[cuda]==0.4.8 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install glew & others
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
export CPATH=$CONDA_PREFIX/include
pip install patchelf

# Recover versions
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
pip install numpy==1.23.0
pip install scipy==1.10.1
```
Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Pretrain world model

For training the world model, we use the training script of [OfflineRL-Kit](https://github.com/yihaosun1124/mobile/tree/main).

For convenience, we provide `run_dynamics.py` that can be utilized to train the model with OfflineRL-Kit.
```bash
cd ..
git clone https://github.com/yihaosun1124/OfflineRL-Kit
cd OfflineRL-Kit
python setup.py install
cp ../LEQ/dynamics/run_dynamics.py run_example/run_dynamics.py
cp -r ../LEQ/d4rl_ext .
```

Now, you can train the model with the `run_dynamics.py`. For example, you can run the command as below:
```bash
python run_example/run_dynamics.py --task antmaze-medium-play-v2 --seed 3
```

### Run training

#### LEQ
  antmaze-umaze-v2
  antmaze-umaze-diverse-v2
  antmaze-medium-play-v2
  antmaze-medium-diverse-v2
  antmaze-large-play-v2
  antmaze-large-diverse-v2
  hopper-random-v2
  hopper-medium-v2
  hopper-medium-replay-v2
  hopper-medium-expert-v2
  walker2d-random-v2
  walker2d-medium-v2
  walker2d-medium-replay-v2
  walker2d-medium-expert-v2
  halfcheetah-random-v2
  halfcheetah-medium-v2
  halfcheetah-medium-replay-v2
  halfcheetah-medium-expert-v2
```bash
PYTHONPATH='.' python train/train_LEQ.py --env_name=walker2d-medium-replay-v2 --expectile 0.5
PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-medium-play-v2 --expectile 0.5
PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-medium-diverse-v2 --expectile 0.4

PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-umaze-v2 --expectile 0.3 --seed 3
PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-umaze-diverse-v2 --expectile 0.3 --seed 3
PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-medium-play-v2 --expectile 0.3 --seed 3
PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-medium-diverse-v2 --expectile 0.3 --seed 3
PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-large-play-v2 --expectile 0.3 --seed 3
PYTHONPATH='.' python train/train_LEQ.py --env_name=antmaze-large-diverse-v2 --expectile 0.3 --seed 3
```

#### MOBILEQ (Please refer to the ablation study section of the paper for details)

```bash
PYTHONPATH='.' python train/train_MOBILEQ.py --env_name=Hopper-v3-medium --beta 1.0
```

#### MOBILE (Jax implementation of [Sun et al.](https://github.com/yihaosun1124/mobile/tree/main))

```bash
PYTHONPATH='.' python train/train_MOBILE.py --env_name=antmaze-large-play-v2 --beta 1.0
```

## References

* The implementation is based on [IQL](https://github.com/ikostrikov/implicit_q_learning/).
* MOBILE implementation is from [OfflineRLKit](https://github.com/yihaosun1124/OfflineRL-Kit).



