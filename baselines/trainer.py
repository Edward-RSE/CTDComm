from collections import namedtuple
from inspect import getfullargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=args.alpha, eps=args.eps)  # don't know why these weren't input args #alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        if torch.cuda.is_available():
            self.set_device("cuda")
        else:
            self.set_device("cpu")

    def set_device(self, device):
        self.device = device
        self.policy_net.set_device(device)
        self.policy_net = self.policy_net.to(device)
        torch.set_default_device(self.device)

    def get_episode(self, epoch):
        episode = []
        reset_args = getfullargspec(self.env.reset).args
        if self.args.env_name == 'dec_predator_prey':
            observations, info = self.env.reset(self.args.env_seed)
            # Convert observation dict into a 'state' array for backwards compatibility (batch size=1)
            state = np.stack([obs.flatten() for _, obs in observations.items()], dtype=np.double)
            state = np.expand_dims(state, 0)
            state = torch.tensor(state).to(self.device)
            # flattened_obs = [
            #     torch.from_numpy(obs.flatten()) for obs in observations.values()
            # ]
            # stacked_obs = torch.stack(flattened_obs, dim=0).double()
            # state = stacked_obs.unsqueeze(0)
        else:
            if 'epoch' in reset_args:
                state = self.env.reset(epoch)
            else:
                state = self.env.reset()
            info = dict()
        should_display = self.display and self.last_step

        if should_display:
            # None of the envs have a 'display', so I presume they mean render - JenniBN
            self.env.render()
        stat = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        if self.args.save_adjacency:
            episode_adjacency = []

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info["comm_action"] = torch.zeros(self.args.nagents, dtype=int)
            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]

                if self.args.gacomm and self.args.save_adjacency:
                    action_out, value, prev_hid, comm_density, timestep_adjacency = self.policy_net(x, info)
                elif self.args.gacomm:
                    action_out, value, prev_hid, comm_density = self.policy_net(x, info)
                elif self.args.save_adjacency:
                    action_out, value, prev_hid, timestep_adjacency = self.policy_net(x, info)
                else:
                    action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if self.args.gacomm and self.args.save_adjacency:
                    action_out, value, comm_density, timestep_adjacency = self.policy_net(x, info)
                elif self.args.gacomm:
                    action_out, value, comm_density = self.policy_net(x, info)
                elif self.args.save_adjacency:
                    action_out, value, timestep_adjacency = self.policy_net(x, info)
                else:
                    action_out, value, = self.policy_net(x, info)

            # Save adjacency data for network analysis
            if self.args.save_adjacency:
                episode_adjacency.append(timestep_adjacency)


            if self.args.gacomm:
                stat['density1'] = stat.get('density1', 0) + comm_density[0]
                stat['density2'] = stat.get('density2', 0) + comm_density[1]

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            if 'dec' in self.args.env_name:
                # Building in forwards compatibility with the PettingZoo/AEC API
                #TODO: does the dict setup still work when learning agents die?
                actual = {self.env.possible_agents[i]: actual[0][i] for i in range(len(self.env.possible_agents))}
                next_state, rewards, terminations, truncations, infos = self.env.step(actual)
                reward = torch.tensor(list(rewards.values()))
                terminated = np.all(list(terminations.values()))
                truncated = np.all(list(truncations.values()))
                done = terminated or truncated

                info = {
                    "agent_locs": self.env.locs,
                    "alive_mask": torch.ones_like(reward),
                }
                for agent, info_val in infos.items():
                    # The environment uses a string to explain that the agent is dead/inactive
                    if isinstance(info_val, str):
                        info['alive_mask'][agent[1]] = 0

                # Convert next_state dict into a 'state' array for backwards compatibility (batch size=1)
                next_state = np.stack([obs.flatten() for _, obs in next_state.items()])
                next_state = np.expand_dims(next_state, 0)
                next_state = torch.from_numpy(next_state).double().to(self.device)
            else:
                next_state, reward, done, info = self.env.step(actual)

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info["comm_action"] = (
                    action[-1]
                    if not self.args.comm_action_one
                    else torch.ones(self.args.nagents, dtype=int)
                )

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc["alive_mask"] = torch.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm \
                        or (hasattr(self.args, 'learning_prey') and self.args.learning_prey):
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = torch.ones(reward.shape)
            episode_mini_mask = torch.ones(reward.shape)

            if done:
                episode_mask = torch.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.render()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            rewards = self.env.reward_terminal()
            if 'dec' in self.args.env_name:
                reward = torch.tensor(list(rewards.values()))

            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if (hasattr(self.args, 'enemy_comm') and self.args.enemy_comm) \
                or (hasattr(self.args, 'learning_prey') and self.args.learning_prey):
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)

        if self.args.save_adjacency:
            return episode, stat, np.array(episode_adjacency)
        else:
            return episode, stat

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.stack(batch.reward)
        episode_masks = torch.stack(batch.episode_mask)
        episode_mini_masks = torch.stack(batch.episode_mini_mask)
        # actions = torch.tensor(batch.action)
        actions = torch.stack([torch.stack(a) for a in batch.action])
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.concatenate(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]
        alive_masks = torch.concatenate([item["alive_mask"] for item in batch.misc])

        coop_returns = torch.zeros(batch_size, n, device=self.device)
        ncoop_returns = torch.zeros(batch_size, n, device=self.device)
        returns = torch.zeros(batch_size, n, device=self.device)
        deltas = torch.zeros(batch_size, n, device=self.device)
        advantages = torch.zeros(batch_size, n, device=self.device)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        # Leaving this here in case I decide I want batch adjacency again, but it's a lot of unwieldy data tbh
        # For now, just save the adjacency for the last episode of the batch (arbitrary but easiest)
        # if self.args.save_adjacency:
        #     batch_adjacency = []
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            if self.args.save_adjacency:
                episode, episode_stat, episode_adjacency = self.get_episode(epoch)
                # batch_adjacency.append(episode_adjacency)
            else:
                episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        if self.args.save_adjacency:
            return batch, self.stats, episode_adjacency #np.array(batch_adjacency)
        else:
            return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        if self.args.save_adjacency:
            batch, stat, batch_adjacency = self.run_batch(epoch)
        else:
            batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        if self.args.save_adjacency:
            return stat, batch_adjacency
        else:
            return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
