import os
import json

import numpy as np
import HCNN_agent as Agent
import env4training as Env

NUM_EPISODES = 10
DIR = os.path.join('results', 'HCNN')

env = Env.BlockTrainingEnvironment()
agent = Agent.CNNAgent(grid_sizes=env.grid_sizes, 
                       num_orient=env.num_orient, 
                       block_info_size=env.block_info_size, 
                       block_types=env.block_types)
log = {
    "record":[]
}


def log_everything(e, block, agent_log, reward, terminal):
    event = {
        "episode": int(e),
        "block": int(block),
        "agent": agent_log,
        "reward": float(reward),
        "terminal": bool(terminal)
    }
    log["record"].append(event)

for ep in range(NUM_EPISODES):
    print(f'==EPISODE {ep}==')

    state, reward, terminal = env.reset()

    while not terminal:
        # print('AGENT selects ACTION')
        agent_actions = agent.select_actions(state)

        # print('ENV makes STEP')
        next_state, reward, terminal = env.step(agent_actions)

        # print('AGENT updates EXP')
        agent.update_experience(state,agent_actions,next_state,reward,terminal)

        log_everything(ep, env.block_seq_index, agent.log, reward, terminal)
        state = next_state
    
    print(f'==END EPISODE {ep}==')
    print()
    print()

    agent.update_policy(ep+1)

    log_file = os.path.join(DIR,f'episode_{ep}_log.json')

    with open(log_file, 'w') as f:
        print("Saving log...")
        print()
        print()
        json.dump(log, f)

    # Clear log
    log = {
        "record":[]
    }