import os
import json
import copy
import datetime

import HCNN_agent as Agent
import env4training as Env

NUM_EPISODES = 2
DIR = os.path.join('results', 'HCNN')

START_EP = 0
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

check_every = 50

if LOAD_CHECKPOINT:
    agent.load_checkpoint()
    START_EP = agent.episode + 1

start_time = datetime.datetime.now()

for ep in range(NUM_EPISODES):
    episode = START_EP + ep
    start_block = 0
    check_block = check_every
    print(f'===== EPISODE {episode} =====')

    state, reward, terminal = env.reset()
    
    print('== PLACING BLOCKS ==')
    while not terminal:
        # print('= AGENT selects ACTION =')
        agent_actions, env_actions = agent.select_actions(state)

        # print('= ENV makes STEP =')
        next_state, reward, terminal = env.step(env_actions)

        # print('= AGENT updates EXP =')
        loss = agent.update_experience(state,agent_actions,next_state,reward,terminal)

        log_everything(episode, env.block_seq_index, env.log, agent.log, loss, reward, terminal)

        # print(env.block_seq_index)
        if ((env.block_seq_index+1) >= check_block) or terminal:
            print(f'{datetime.datetime.now()} TRAINING: Block {env.block_seq_index}, Reward = {reward:.2f}, Percent complete = {env.perc_complete*100:.2f}%')
            # print('!!!!SAVE!!!!')
            agent.save_checkpoint(loss)
            log_file = os.path.join(DIR,f'episode_{episode}_blocks_{start_block}-{env.block_seq_index}_log.json')
            start_block = env.block_seq_index+1
            check_block += check_every

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
    
    print()
    print(f'TRAINING TIME ELAPSED: {datetime.datetime.now() - start_time}')
    print(f'===== END EPISODE {episode} =====')
    print()
    print()

    agent.update_policy(episode+1)   