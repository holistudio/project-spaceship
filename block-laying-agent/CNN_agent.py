import torch
import torch.nn as nn

import numpy as np

import os
import random
import math
from collections import namedtuple, deque

DIR = os.path.join('results', 'CNN')
PATH = os.path.join(DIR, 'CNN_checkpoint.tar')

n_h = 32

BATCH_SIZE = 32 # 128
LR = 1e-3 # 3e-4

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 93944

GAMMA = 0.99

TAU = 0.005

UPDATE_TARGET_EP = 5

torch.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class CNN_DQN(nn.Module):
    def __init__(self, grid_sizes, num_orient, block_info_size, block_types, n_hidden, dropout):
        super().__init__()
        self.num_x, self.num_y, self.num_z = grid_sizes
        self.num_orient = num_orient
        self.block_types = block_types
        self.n_hidden = n_hidden
        self.conv1 = nn.Conv3d(in_channels=block_info_size, out_channels=block_types*num_orient, kernel_size=1)
        # self.conv1 = nn.Conv3d(in_channels=block_info_size, out_channels=n_hidden, kernel_size=1)
        # self.fnn = nn.Sequential(nn.Linear(n_hidden, n_hidden),
        #                          nn.ReLU(),
        #                          nn.Linear(n_hidden, n_hidden),
        #                          nn.Dropout(dropout))
        # self.conv2 = nn.Conv3d(in_channels=n_hidden, out_channels=block_types*num_orient, kernel_size=1)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv3d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        (N, C, D, H, W) = x.shape
        x = self.conv1(x)
        # x = self.fnn(x.view(N*D*H*W, self.n_hidden))
        # x = torch.reshape(x, (N, self.n_hidden, D, H, W))
        # x = self.conv2(x)
        x = torch.reshape(x, (N, self.block_types, self.num_orient, self.num_x, self.num_y, self.num_z))
        return x

class CNNAgent(object):
    def __init__(self, grid_sizes, num_orient, block_info_size, block_types):
        self.action_space = [block_types, num_orient, grid_sizes[0],grid_sizes[1],grid_sizes[2]]
        self.num_actions = block_types*num_orient*grid_sizes[0]*grid_sizes[1]*grid_sizes[2]
        self.agent_actions = torch.tensor([[0]], device=device, dtype=torch.long)

        self.policy_net = CNN_DQN(grid_sizes, num_orient, block_info_size, block_types, n_hidden=n_h, dropout=0.2).to(device)
        self.target_net = CNN_DQN(grid_sizes, num_orient, block_info_size, block_types, n_hidden=n_h, dropout=0.2).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.episode = 0
        self.steps_done = 0

        self.memory = ReplayMemory(capacity=1000)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

        self.log = {
            "actions": [0,0,0,0,0],
            "explore_exploit": "explore",
            "epsilon": 0.0,
            "eps_steps": 0
        }

    def interpret_agent_actions(self, agent_actions):
        max_index = agent_actions.squeeze().item()
        env_actions = np.unravel_index(max_index,(self.action_space))
        block_type_i, orientation, grid_x, grid_y, grid_z = env_actions
        return block_type_i, orientation, grid_x, grid_y, grid_z
    
    def select_actions(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1
        # print(f'Epsilon={eps_threshold}')
        if sample > eps_threshold:
            # print('Agent EXPLOITS!')
            mode = "exploit"
            
            batch_state = state.unsqueeze(0).float()
            with torch.no_grad():
                out = self.policy_net(batch_state)
                out = out.squeeze()

                # Find the index of the maximum value
                max_index = torch.argmax(out)

                # Convert the flat index to multidimensional indices
                # indices = []
                # for dim_size in reversed(out.shape):
                #     indices.append((max_index % dim_size).item())
                #     max_index //= dim_size

                # # Reverse the list of indices to match the tensor's shape
                # indices.reverse()
                # indices = np.unravel_index(max_index.item(), out.shape)
        else:
            # print('Agent EXPLORES!')
            mode = "explore"

            # indices = [random.randint(0,a-1) for a in self.action_space]
            max_index = random.randint(0,self.num_actions-1) 
        
        self.agent_actions[0,0] = max_index

        self.log = {
            "actions": [int(max_index)],
            "explore_exploit": mode,
            "epsilon": eps_threshold,
            "eps_steps": self.steps_done
        }

        block_type_i, orientation, grid_x, grid_y, grid_z = self.interpret_agent_actions(self.agent_actions)

        return self.agent_actions, (block_type_i, orientation, grid_x, grid_y, grid_z)
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return -1
        
        # print('Sample memory')
        transitions = self.memory.sample(BATCH_SIZE)

        # print('Transpose the batch')
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # print('Non-final mask')
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(device)
        non_final_batches = non_final_next_states.shape[0]
        
        # print('Make batches')
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # print('Compute state_action_values')
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values = self.policy_net(state_batch)[:, 
        #                                                    action_batch[:,0], 
        #                                                    action_batch[:,1], 
        #                                                    action_batch[:,2], 
        #                                                    action_batch[:,3], 
        #                                                    action_batch[:,4]][0].reshape((BATCH_SIZE,1))
        state_action_values = self.policy_net(state_batch).view(BATCH_SIZE,-1).gather(1, action_batch)
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # print('Compute next state values')
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            batch_values = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = torch.max(batch_values.view(non_final_batches,-1),dim=-1).values
            # for i in range(len(batch_values)):
                # out = batch_values[i]
                # max_index = torch.argmax(out)

                # Convert the flat index to multidimensional indices
                # indices = []
                # for dim_size in reversed(out.shape):
                #     indices.append((max_index % dim_size).item())
                #     max_index //= dim_size

                # Reverse the list of indices to match the tensor's shape
                # indices.reverse()
                # indices = np.unravel_index(max_index.item(), out.shape)
                
                # next_state_values[non_final_mask[i]] = out[indices[0],indices[1],indices[2],indices[3],indices[4]].item()
            # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # print('Compute Q-values')
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # print('Compute Huber loss')
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # print('Backprop')
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # print('Gradient clipping')
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        # print('Optimizer step')
        self.optimizer.step()

        return loss
        
    def soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def update_experience(self, state, agent_actions, next_state, reward, terminal):
        state = state.unsqueeze(0).float().cpu().detach().clone()

        if terminal:
            next_state = None
        else:
            next_state = next_state.unsqueeze(0).float().cpu().detach().clone()
        reward = torch.tensor([reward]).cpu()

        # print('AGENT pushes to MEMORY')
        self.memory.push(state, agent_actions.cpu().detach().clone(), next_state, reward)

        # print('AGENT OPTIMIZES')
        loss = self.optimize_model()

        # print('AGENT soft UPDATES')
        self.soft_update()

        return loss

    def update_policy(self, episode):
        if episode % UPDATE_TARGET_EP == 0:
            print('Setting target_net to policy_net...')
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print()

    def save_checkpoint(self, loss):
        torch.save({
            'episode': self.episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_memory': self.memory,
            'loss': loss,
            'batch_size': BATCH_SIZE,
            'eps_start': EPS_START,
            'eps_end': EPS_END,
            'eps_decay': EPS_DECAY,
            'eps_steps': self.steps_done,
            'gamma': GAMMA,
            'tau': TAU,
            }, PATH)
        return
    
    def load_checkpoint(self):
        checkpoint = torch.load(PATH)
        self.episode = checkpoint['episode']

        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.steps_done = checkpoint['eps_steps']

        for mem in checkpoint['replay_memory'].memory:
            self.memory.push(mem[0], mem[1], mem[2], mem[3])
        return