import gym
import numpy as np
import torch
from DDPG import DDPG_Agent, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import str2bool,evaluate_policy


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='PV0, Lch_Cv2, Humanv2, HCv2, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=5e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--warmup', type=int, default=5e4, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')
opt = parser.parse_args()
print(opt)



def main():
    EnvName = ['Pendulum-v0','LunarLanderContinuous-v2','Humanoid-v2','HalfCheetah-v2','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV0', 'LLdV2', 'Humanv2', 'HCv2','BWv3', 'BWHv3']

    # Build Env
    env = gym.make(EnvName[opt.EnvIdex])
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    print('Env:', EnvName[opt.EnvIdex], '  state_dim:', opt.state_dim, '  action_dim:', opt.action_dim, '  max_a:', opt.max_action,
          '  min_a:', env.action_space.low[0],'  max_e_steps:',env._max_episode_steps )

    # Random seed config:
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # Build SummaryWriter to record training curves
    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    model = DDPG_Agent(opt)
    if opt.Loadmodel: model.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        score = evaluate_policy(eval_env, model, turns=10, render=True)
        print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, done, ep_steps = env.reset(), False, 0

            '''Interact & trian'''
            while not done:  
                if total_steps < opt.warmup:
                    a = env.action_space.sample()
                else:
                    a = (model.select_action(s) + np.random.normal(0, opt.max_action * opt.noise, size=opt.action_dim) 
                         ).clip(-opt.max_action, opt.max_action)  # explore: deterministic actions + noise
                s_next, r, done, info = env.step(a)
                ep_steps += 1 #steps of current episode

                '''Avoid impacts caused by reaching max episode steps'''
                if (done and ep_steps != env._max_episode_steps):
                    dw = True  # dw: dead and win, namely terminated
                else:
                    dw = False # truncated

                model.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                # train the model
                if total_steps >= opt.warmup:
                    model.train()   

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, model)              
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:', BrifEnvName[opt.EnvIdex], 'steps: {}k'.format(int(total_steps/1000)), 'score:', score)
                         

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    model.save(BrifEnvName[opt.EnvIdex],total_steps)
        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()




