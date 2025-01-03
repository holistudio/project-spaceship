import torch
import torch.nn as nn

import numpy as np

import os
import random
import math
from collections import namedtuple, deque

DIR = os.path.join('results', 'HCNN')
PATH = os.path.join(DIR, 'HCNN_checkpoint.tar')

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
    
class HCNN_DQN(nn.Module):
    def __init__(self, grid_sizes, num_orient, block_info_size, block_types, dropout):
        super().__init__()
        self.num_x, self.num_y, self.num_z = grid_sizes
        self.num_orient = num_orient
        self.block_types = block_types

        D = self.num_z
        D_out = 4
        padding = 0
        dilation = 1
        stride = 1

        kernel_size = int((((D_out - 1)*stride - D + 1 - 2*padding)/(-dilation)) + 1)

        self.conv_high = nn.Conv3d(in_channels=block_info_size, out_channels=1, kernel_size=kernel_size)
        self.conv_med = nn.Conv3d(in_channels=block_info_size, out_channels=1, kernel_size=kernel_size)
        self.conv_low = nn.Conv3d(in_channels=block_info_size, out_channels=1, kernel_size=kernel_size)
        self.fnn_block = nn.Sequential(
            nn.Linear(D_out*D_out*D_out, 4 * D_out*D_out*D_out),
            nn.ReLU(),
            nn.Linear(4 * D_out*D_out*D_out, block_types),
            nn.Dropout(dropout),
        )
        self.fnn_orient = nn.Sequential(
            nn.Linear(D_out*D_out*D_out, 2 * D_out*D_out*D_out),
            nn.ReLU(),
            nn.Linear(2 * D_out*D_out*D_out, num_orient),
            nn.Dropout(dropout),
        )

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
        out_high = self.conv_high(x)
        out_med = self.conv_med(x)
        out_low = self.conv_low(x)

        out_block = self.fnn_block(out_low.view(N,-1))
        out_orient = self.fnn_orient(out_low.view(N,-1))

        return out_block, out_orient, out_high, out_med, out_low

class CNNAgent(object):
    def __init__(self, grid_sizes, num_orient, block_info_size, block_types):
        self.action_space = [block_types, num_orient, grid_sizes[0],grid_sizes[1],grid_sizes[2]]
        self.num_actions = block_types*num_orient*grid_sizes[0]*grid_sizes[1]*grid_sizes[2]
        self.agent_actions = torch.zeros(5, device=device, dtype=torch.long).unsqueeze(0)

        self.policy_net = HCNN_DQN(grid_sizes, num_orient, block_info_size, block_types, dropout=0.2).to(device)
        self.target_net = HCNN_DQN(grid_sizes, num_orient, block_info_size, block_types, dropout=0.2).to(device)
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
        max_indices = agent_actions.squeeze()

        block_type_i = max_indices[0].item()
        orientation = max_indices[1].item()

        high_i = np.array(np.unravel_index(max_indices[2].item(),(4,4,4)))
        med_i = np.array(np.unravel_index(max_indices[3].item(),(4,4,4)))
        low_i = np.array(np.unravel_index(max_indices[4].item(),(4,4,4)))
        grid_x, grid_y, grid_z = high_i*int(self.action_space[4]/4) + med_i*int(self.action_space[4]/(4*4)) + low_i

        return block_type_i, orientation, grid_x, grid_y, grid_z
    
    def select_actions(self, state):
        sample = random.random()
        mode = ""

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        # eps_threshold = 0

        self.steps_done += 1

        # print(f'Epsilon={eps_threshold}')
        if sample > eps_threshold:
            # print('Agent EXPLOITS!')
            mode = "exploit"

            batch_state = state.unsqueeze(0).float()
            with torch.no_grad():
                out_block, out_orient, out_high, out_med, out_low = self.policy_net(batch_state)

                # Find the index of the maximum value
                self.agent_actions[0,0] = torch.argmax(out_block).item()
                self.agent_actions[0,1] = torch.argmax(out_orient).item()

                self.agent_actions[0,2] = torch.argmax(out_high).item()
                self.agent_actions[0,3] = torch.argmax(out_med).item()
                self.agent_actions[0,4] = torch.argmax(out_low).item()
        else:
            # print('Agent EXPLORES!')
            mode = "explore"

            # indices = [random.randint(0,a-1) for a in self.action_space]
            self.agent_actions[0,0] = torch.randint(0,self.action_space[0],(1,)).item()
            self.agent_actions[0,1] = torch.randint(0,self.action_space[1],(1,)).item()

            self.agent_actions[0,2] = torch.randint(0,4*4*4,(1,)).item()
            self.agent_actions[0,3] = torch.randint(0,4*4*4,(1,)).item()
            self.agent_actions[0,4] = torch.randint(0,4*4*4,(1,)).item()

        self.log = {
            "actions": list(map(int, self.agent_actions.clone().squeeze().numpy(force=True))),
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
        
        # state_action_values = self.policy_net(state_batch).view(BATCH_SIZE,-1).gather(1, action_batch)
        out_block, out_orient, out_high, out_med, out_low = self.policy_net(state_batch)
        q_block = out_block.view(BATCH_SIZE,-1).gather(1, action_batch[:,0].unsqueeze(1))
        q_orient = out_orient.view(BATCH_SIZE,-1).gather(1, action_batch[:,1].unsqueeze(1))
        q_high = out_high.view(BATCH_SIZE,-1).gather(1, action_batch[:,2].unsqueeze(1))
        q_med = out_med.view(BATCH_SIZE,-1).gather(1, action_batch[:,3].unsqueeze(1))
        q_low = out_low.view(BATCH_SIZE,-1).gather(1, action_batch[:,4].unsqueeze(1))
        state_action_values = torch.cat((q_block,q_orient,q_high,q_med,q_low),dim=1)
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # print('Compute next state values')
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((BATCH_SIZE,5), device=device)
        with torch.no_grad():
            v_block, v_orient, v_high, v_med, v_low = self.target_net(non_final_next_states)
            v_block = torch.max(v_block.view(non_final_batches,-1),dim=-1).values.unsqueeze(1)
            v_orient = torch.max(v_orient.view(non_final_batches,-1),dim=-1).values.unsqueeze(1)
            v_high = torch.max(v_high.view(non_final_batches,-1),dim=-1).values.unsqueeze(1)
            v_med = torch.max(v_med.view(non_final_batches,-1),dim=-1).values.unsqueeze(1)
            v_low = torch.max(v_low.view(non_final_batches,-1),dim=-1).values.unsqueeze(1)
            next_state_values[non_final_mask] = torch.cat((v_block,v_orient,v_high,v_med,v_low),dim=1)
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
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(1)
        # expected_state_action_values = expected_state_action_values.unsqueeze(1)

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
        self.episode = episode
        if self.episode % UPDATE_TARGET_EP == 0:
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
    