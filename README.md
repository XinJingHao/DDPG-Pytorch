# DDPG-Pytorch
A clean Pytorch implementation of DDPG on continuous action space. Here is the result (all the experiments are trained with same hyperparameters):  

Pong| Enduro
:-----------------------:|:-----------------------:|
<img src="https://github.com/XinJingHao/DQN-DDQN-Atari-Pytorch/raw/main/IMGs/Pong.png" width="320" height="200">| <img src="https://github.com/XinJingHao/DQN-DDQN-Atari-Pytorch/raw/main/IMGs/Enduro.png" width="320" height="200">

Note that DDPG is notoriously susceptible to hyperparameters and thus is unstable sometimes. We strongly recommend you use its refinement [TD3](https://github.com/XinJingHao/TD3-Pytorch).
**Other RL algorithms by Pytorch can be found [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch).**

## Dependencies
```bash
gym==0.19.0  
box2d==2.3.8  
numpy==1.21.6  
pytorch==1.11.0 
tensorboard==2.9.1
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
