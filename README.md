# DDPG-Pytorch
A clean Pytorch implementation of DDPG on continuous action space. Here is the result (all the experiments are trained with same hyperparameters):  

Pendulum| LunarLanderContinuous
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/DDPG-Pytorch/blob/main/IMGs/ddpg_pv0.svg" width="320" height="200">| <img src="https://github.com/XinJingHao/DDPG-Pytorch/blob/main/IMGs/ddpg_lld.svg" width="320" height="200">

Note that DDPG is notoriously susceptible to hyperparameters and thus is unstable sometimes. We strongly recommend you use its refinement [TD3](https://github.com/XinJingHao/TD3-Pytorch).
**Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**

## Dependencies
```bash
python == 3.8.5
gym == 0.19.0  
pyglet == 1.5.15
box2d == 2.3.10  
numpy == 1.24.3  
pytorch == 1.13.1
tensorboard == 2.13.0
```

## How to use my code
### Train from scratch
```bash
python main.py
```
where the default enviroment is Pendulum-v0.  
### Change Enviroment
If you want to train on different enviroments, just run 
```bash
python main.py --EnvIdex 1
```
The --EnvIdex can be set to be 0~5, where
```bash
'--EnvIdex 0' for 'Pendulum-v0'  
'--EnvIdex 1' for 'LunarLanderContinuous-v2'  
'--EnvIdex 2' for 'Humanoid-v2'  
'--EnvIdex 3' for 'HalfCheetah-v2'  
'--EnvIdex 4' for 'BipedalWalker-v3'  
'--EnvIdex 5' for 'BipedalWalkerHardcore-v3' 
```

P.S. if you want train on 'Humanoid-v2' or 'HalfCheetah-v2', you need to install **MuJoCo** first.
### Play with trained model
```bash
python main.py --EnvIdex 0 --write False --render True --Loadmodel True --ModelIdex 100
```
which will render the 'Pendulum-v0'.  
### Visualize the training curve
You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'
### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'
### Reference
DDPG: [Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[J]. arXiv preprint arXiv:1509.02971, 2015.](https://arxiv.org/abs/1509.02971)
