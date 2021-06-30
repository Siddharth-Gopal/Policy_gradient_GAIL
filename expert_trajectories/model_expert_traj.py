import torch
import gym
import keyboard
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.actor_net import ActorNet
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('-m', 'model_path', required=True, metavar='', help='name of model that is present in the same dir')
# parser.add_argument('-s', 'traj_save_path',required=True, metavar='', help='output npz file of trajectory data')
# args = parser.parse_args()

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = os.path.join('PPO', 'ppo_500.pt')
policy_net = ActorNet(4,2)
policy_net.to(Device)
policy_net.load_state_dict(torch.load(model_path))
policy_net.eval()

# While choosing actions we will simply choose the maximum in our action space
def action_chosen(obs):
    state = torch.as_tensor(obs).float().to(Device)
    logits = policy_net(state).view(-1)
    action = torch.argmax(logits)
    return int(action)


env = gym.make('CartPole-v1')
render = False
obs = []
actions = []
rewards = []
dones = []
max_eps = 8
max_steps = 800
for eps in range(max_eps):
    observation = env.reset()
    ep_legnth = 0
    for steps in range(max_steps):
        # env.render()
        obs.append(observation)
        action = action_chosen(observation)
        # action = env.action_space.sample()

        
        observation, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        ep_legnth += 1
        if done:
            print('this ep over. ep legnth: ', ep_legnth)
            break

obs = np.array(obs)
actions = np.array(actions)
rewards = np.array(rewards)
dones = np.array(dones)


# print('observations: ', obs)
# print('sizes: ', obs.shape)
# print(actions)
# print(rewards)
# print(dones)
# print(ep_ls)
print()

np.savez(os.path.join('GAIL', 'expert_trajectories','expert_traj_test.npz'), \
    obs = obs, acts=actions, rews=rewards, dones=dones)

env.close()