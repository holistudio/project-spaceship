import os
import json
import copy

import HCNN_agent as Agent
import env4training as Env

NUM_EPISODES = 1
DIR = os.path.join('results', 'HCNN')

LOAD_CHECKPOINT = False

env = Env.BlockTrainingEnvironment()
agent = Agent.CNNAgent(grid_sizes=env.grid_sizes, 
                       num_orient=env.num_orient, 
                       block_info_size=env.block_info_size, 
                       block_types=env.block_types)

log = {
    "record":[]
}


def log_everything(e, block, env_log, agent_log, loss, reward, terminal):
    event = {
        "episode": int(e),
        "block": int(block),
        "env": copy.deepcopy(env_log),
        "agent": copy.deepcopy(agent_log),
        "loss": float(loss),
        "reward": float(reward),
        "terminal": bool(terminal)
    }
    log["record"].append(event)

episode = 0
block = 0

if LOAD_CHECKPOINT:
    agent.load_checkpoint()
    episode = agent.episode + 1

for ep in range(NUM_EPISODES):
    episode = episode + ep
    print(f'==EPISODE {episode}==')

    state, reward, terminal = env.reset()

    while not terminal:
        # print('AGENT selects ACTION')
        agent_actions, env_actions = agent.select_actions(state)

        # print('ENV makes STEP')
        next_state, reward, terminal = env.step(env_actions)

        # print('AGENT updates EXP')
        loss = agent.update_experience(state,agent_actions,next_state,reward,terminal)

        log_everything(episode, env.block_seq_index-1, env.log, agent.log, loss, reward, terminal)

        if ((env.block_seq_index-1) % 50 == 0) or terminal:
            agent.save_checkpoint(loss)
            if (env.block_seq_index-1) != 0:
                log_file = os.path.join(DIR,f'episode_{episode}_blocks_{block}-{block+50}_log.json')
                block += 50

                with open(log_file, 'w') as f:
                    # print("Saving log...")
                    # print()
                    # print()
                    json.dump(log, f)

                # Clear log
                log = {
                    "record":[]
                }
            
        state = next_state
    
    print(f'==END EPISODE {episode}==')
    print()
    print()

    agent.update_policy(episode+1)   