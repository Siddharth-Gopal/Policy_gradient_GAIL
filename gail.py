import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import gym
import os
import sys
from statistics import mean
import matplotlib.pyplot as plt 
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.actor_net import ActorNet
from models.critic_net import CriticNet
from models.discriminator import Discriminator

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PGAgent():
    def __init__(self):
        
        self.env = gym.make('CartPole-v1')
        self.expert_traj_fpath = os.path.join('GAIL', 'expert_trajectories', 'expert_traj_test.npz')
        self.save_policy_fpath = os.path.join('GAIL','gail_actor1.pt')
        self.save_rewards_fig = os.path.join('GAIL', 'gail_rewards.png')
        #
        self.state_space = 4
        self.action_space = 2

        # These values are taken from the env's state and action sizes
        self.actor_net = ActorNet(self.state_space, self.action_space)
        self.actor_net.to(device=Device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr = 0.0001)

        self.critic_net = CriticNet(self.state_space)
        self.critic_net.to(Device)
        self.critic_net_optim = torch.optim.Adam(self.critic_net.parameters(), lr = 0.0001)

        self.discriminator = Discriminator(self.state_space, self.action_space)
        self.discriminator.to(Device)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr = 0.0001)

        # Storing all the values used to calculate the model losses
        self.traj_obs = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_dones = []
        self.traj_logprobs = []
        self.traj_logits = []
        self.traj_state_values = []

        # Discount factor
        self.gamma = 0.95
        # Bias Variance tradeoff (higher value results in high variance, low bias)
        self.gae_lambda = 0.95

        # These two will be used during the training of the policy
        self.ppo_batch_size = 500
        self.ppo_epochs = 12
        self.ppo_eps = 0.2

        # Discriminator
        self.num_expert_transitions = 150

        # These will be used for the agent to play using the current policy
        self.max_eps = 5
        # Max steps in mountaincar ex is 200
        self.max_steps = 800

        # documenting the stats
        self.avg_over = 5 # episodes
        self.stats = {'episode': 0, 'ep_rew': []}

    def clear_lists(self):
        self.traj_obs = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_dones = []
        self.traj_logprobs = []
        self.traj_logits = []
        self.traj_state_values = []

    # This returns a categorical torch object, which makes it easier to calculate log_prob, prob and sampling from them
    def get_logits(self, state):
        logits = self.actor_net(state)
        return Categorical(logits=logits)

    def calc_policy_loss(self, states, actions, rewards):
        assert (torch.is_tensor(states) and torch.is_tensor(actions) and torch.is_tensor(rewards)),\
             "states and actions are not in the right format"

        # The negative sign is for gradient ascent
        loss = -(self.get_logits(states).log_prob(actions))*rewards
        return loss.mean()

    def ppo_calc_log_prob(self, states, actions):
        obs_tensor = torch.as_tensor(states).float().to(device=Device)
        actions = torch.as_tensor(actions).float().to(device=Device)
        logits = self.get_logits(obs_tensor)
        entropy = logits.entropy()
        log_prob = logits.log_prob(actions)

        return log_prob, entropy

    def get_action(self, state):
        
        # Finding the logits and state value using the actor and critic net
        logits = self.get_logits(state)
        action = logits.sample()

        # Sample in categorical finds probability first and then samples values according to that prob
        return action.item()


        # This gives the reward to go for each transition in the batch
        rew_to_go_list = []
        rew_sum = 0
        for rew, done in zip(reversed(traj_rewards), reversed(traj_dones)):
            if done:
                rew_sum = rew
                rew_to_go_list.append(rew_sum)
            else:
                rew_sum = rew + rew_sum
                rew_to_go_list.append(rew_sum)

        rew_to_go_list = reversed(rew_to_go_list)
        return list(rew_to_go_list)

    # This returns the concatenated state_action tensor used to input to discriminator
    # obs and actions are a list
    def concat_state_action(self, obs_list, actions_list, shuffle=False):
        obs = np.array(obs_list)
        actions_data = np.array(actions_list)
        actions = np.zeros((len(actions_list), self.action_space))
        actions[np.arange(len(actions_list)), actions_data] = 1  # Converting to one hot encoding

        state_action = np.concatenate((obs, actions), axis=1)
        if shuffle:
            np.random.shuffle(state_action)  # Shuffling to break any coorelations

        state_action = torch.as_tensor(state_action).float().to(Device)

        return state_action

    # This uses the discriminator and critic networks to calculate the advantage 
    # of each state action pair, and the targets for the critic network
    def calc_gae_targets(self):

        obs_tensor = torch.as_tensor(self.traj_obs).float().to(Device)
        action_tensor = torch.as_tensor(self.traj_actions).float().to(Device)
        state_action = self.concat_state_action(self.traj_obs, self.traj_actions)

        # This calculates how well we have fooled the discriminator, which is equivalent to the 
        # reward at each time step
        disc_rewards = -torch.log(self.discriminator(state_action))
        disc_rewards = disc_rewards.view(-1).tolist()

        traj_state_values = self.critic_net(obs_tensor).view(-1).tolist()
        gae = []
        targets = []

        for i, val, next_val, reward, done in \
        zip(range(len(self.traj_dones)), \
        reversed(traj_state_values), \
        reversed(traj_state_values[1:] + [None]), \
        reversed(disc_rewards), \
        reversed(self.traj_dones)):

            # last trajectory maybe cut short because we have a limit on max_steps,
            # so last done may not always be True
            if done or i==0: 
                delta = reward - val
                last_gae = delta
            else:
                delta = reward + self.gamma*next_val - val
                last_gae = delta + self.gamma*self.gae_lambda*last_gae
            
            gae.append(last_gae)
            targets.append(last_gae + val)

        return list(reversed(gae)), list(reversed(targets))

    # We use num_transitions samples from expert trajectory, 
    # and all policy trajectory transitions to train discriminator.
    def update_discriminator(self, num_transitions):

        # data stores the expert trajectories used to train the discriminator
        # data is a dictionary with keys: 'obs', 'acts', 'rews', 'dones'
        data = np.load(self.expert_traj_fpath)
        # TODO: Using data is this way blocks a lot of memory. It would be much more efficient to load the numpy
        # array using a generator
        obs = data['obs']
        actions = data['acts']

        # Zeroing Discriminator gradient
        self.discriminator_optim.zero_grad()
        loss = nn.BCELoss()

        ## Sampling from expert trajectories. 
        random_samples_ind = np.random.choice(len(obs), num_transitions) 
        expert_state_action = self.concat_state_action(obs[random_samples_ind], actions[random_samples_ind])

        ## Expert Loss, target for expert trajectory taken 0
        expert_output = self.discriminator(expert_state_action)
        expert_traj_loss = loss(expert_output, torch.zeros((num_transitions, 1), device=Device))
        
        ## Sampling policy trajectories
        policy_state_action = self.concat_state_action(self.traj_obs, self.traj_actions, shuffle=True)

        ## Policy Traj loss, target for policy trajectory taken 1
        policy_traj_output = self.discriminator(policy_state_action)
        policy_traj_loss = loss(policy_traj_output, torch.ones((policy_traj_output.shape[0], 1), device=Device))

        # Updating the Discriminator
        D_loss = expert_traj_loss + policy_traj_loss
        D_loss.backward()
        self.discriminator_optim.step()

    def train_gail(self):

        assert (len(self.traj_obs)==len(self.traj_actions)==len(self.traj_dones)), "Size of traj lists don't match"

        # We use self.num_expert_transitions and all saved policy transitions from play()
        self.update_discriminator(self.num_expert_transitions)
        
        # Finding old log prob.
        # If the number of transistions are too large, this could also be broken down and calculated in batches
        # This uses the traj_states and traj_actions to calculate the log_probs of each action
        with torch.no_grad():
            old_logprob, _ = self.ppo_calc_log_prob(self.traj_obs, self.traj_actions)
            old_logprob.detach()

            traj_gae, traj_targets = self.calc_gae_targets()

        # Performing ppo policy updates in batches
        for epoch in range(self.ppo_epochs):
            for batch_offs in range(0, len(self.traj_dones), self.ppo_batch_size):
                batch_obs = self.traj_obs[batch_offs:batch_offs + self.ppo_batch_size]
                batch_actions = self.traj_actions[batch_offs:batch_offs + self.ppo_batch_size]
                batch_gae = traj_gae[batch_offs:batch_offs + self.ppo_batch_size]
                batch_targets = traj_targets[batch_offs:batch_offs + self.ppo_batch_size]
                batch_old_logprob = old_logprob[batch_offs:batch_offs + self.ppo_batch_size]

                # Zero the gradients
                self.actor_optim.zero_grad()
                self.critic_net_optim.zero_grad()

                # Critic Loss
                batch_obs_tensor = torch.as_tensor(batch_obs).float().to(Device)
                state_vals = self.critic_net(batch_obs_tensor).view(-1)
                batch_targets = torch.as_tensor(batch_targets).float().to(Device)
                critic_loss = F.mse_loss(state_vals, batch_targets)
                
                # Policy and Entropy Loss
                log_prob, entropy = self.ppo_calc_log_prob(batch_obs, batch_actions)
                batch_ratio = torch.exp(log_prob - batch_old_logprob)
                batch_gae = torch.as_tensor(batch_gae).float().to(Device)
                unclipped_objective = batch_ratio * batch_gae
                clipped_objective = torch.clamp(batch_ratio, 1 - self.ppo_eps, 1 + self.ppo_eps) * batch_gae
                policy_loss = -torch.min(clipped_objective, unclipped_objective).mean()
                entropy_loss = -entropy.mean()

                # Performing backprop
                critic_loss.backward()
                # Here both policy_loss and entropy_loss calculate grad values in the actor net.
                # By using retain_graph, the next backward call will add onto the previous grad values.
                policy_loss.backward(retain_graph=True)
                entropy_loss.backward()

                # print('Losses: ', (critic_loss.shape, policy_loss.shape, entropy_loss.shape))
                # print('critic grad values: ', self.critic_net.fc1.weight.grad)
                # print('actor grad values: ', self.actor_net.fc1.weight.grad)

                # Updating the networks
                self.actor_optim.step()
                self.critic_net_optim.step()

    # The agent will play self.max_eps episodes using the current policy, and train on that data
    def play(self, rendering):
        
        self.clear_lists()
        saved_transitions = 0
        for ep in range(self.max_eps):
            obs = self.env.reset()
            ep_reward = 0

            for step in range(self.max_steps):
                
                if rendering==True:
                    self.env.render()

                self.traj_obs.append(obs)
                obs = torch.from_numpy(obs).float().to(device=Device)
                
                # get_action() will run obs through actor network and find the action to take
                action = self.get_action(obs)
                
                # We are saving the reward here, but this will not be used in the optimization of the policy
                # or discriminator, it is only used to track our progress.
                obs, rew, done, info = self.env.step(action)
                ep_reward += rew

                self.traj_actions.append(action)
                self.traj_rewards.append(rew)

                saved_transitions += 1

                if done:
                    # We will not save the last observation, since it is essentially a dead state
                    # This will result in having the same length of obs, action, reward and dones deque
                    self.traj_dones.append(done)
                    self.stats['ep_rew'].append(ep_reward)
                    self.stats['episode'] += 1
                    break

                else:
                    self.traj_dones.append(done)
            # print(f" {ep} episodes over.", end='\r')
            print('episode over. Reward: ', ep_reward)

            
        self.train_gail()


    def run(self, model_path, policy_iterations = 65, show_renders_every = 20, renders = True):
        for i in range(policy_iterations):
            if i%show_renders_every==0:
                self.play(rendering=renders)
            else:
                self.play(rendering=False)
            print(f" Policy updated {i} times")
        
        torch.save(self.actor_net.state_dict(), model_path)
        print('model saved at: ', model_path)

    def plot_rewards(self, avg_over=10):       
        graph_x = np.arange(self.stats['episode'])
        graph_y = np.array(self.stats['ep_rew'])

        assert (len(graph_x) == len(graph_y)), "Plot axes do not match"

        graph_x_averaged = [mean(arr) for arr in np.array_split(graph_x, len(graph_x)/avg_over)]
        graph_y_averaged = [mean(arr) for arr in np.array_split(graph_y, len(graph_y)/avg_over)]

        plt.plot(graph_x_averaged, graph_y_averaged)
        plt.savefig(self.save_rewards_fig)



agent = PGAgent()
agent.run(model_path=agent.save_policy_fpath, policy_iterations = 2000, show_renders_every = 20, renders=False)
agent.plot_rewards()

