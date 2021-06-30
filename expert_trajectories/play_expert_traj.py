## This file is used to store the expert trajectories by playing the game in the openai gym environment from the keyboard.
## However fun this may sound, it can get very exhausting to save expert trajectories by ourselves, which is why I have
## also created another file called load expert traj, which takes in a trained policy net to play and save the trajectories.

import gym
import keyboard
import numpy as np
import os


def action_chosen():
    a = keyboard.read_key()
    if a=='left':
        return 0
    elif a=='right':
        return 1
    else:
        print('Incorrect input, try again!')
        action_chosen()


env = gym.make('CartPole-v1')
obs = []
actions = []
rewards = []
dones = []
ep_ls = []
max_eps = 50
max_steps = 800
for eps in range(2):
    observation = env.reset()
    ep_legnth = 0
    for steps in range(30):
        env.render()
        action = action_chosen()
        # action = env.action_space.sample()

        obs.append(observation)
        observation, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        ep_legnth += 1
        if done:
            print('this ep over: ', done)
            ep_ls.append(ep_legnth)
            break

obs = np.array(obs)
actions = np.array(actions)
rewards = np.array(rewards)
dones = np.array(dones)


print('observations: ', obs)
print('sizes: ', obs.shape)
print(actions)
print(rewards)
print(dones)
print(ep_ls)

np.savez(os.path.join('PPO','expert_traj.npz'), obs = obs, acts=actions, rews=rewards, dones=dones)

env.close()