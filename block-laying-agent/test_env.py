import os
import json
import copy
import datetime

import env4training as Env

NUM_EPISODES = 2
DIR = os.path.join('results', 'env')

START_EP = 0
# LOAD_CHECKPOINT = False

env = Env.BlockTrainingEnvironment()

log = {
    "record":[]
}

def log_everything(e, block, env_log, reward, terminal):
    event = {
        "episode": int(e),
        "block": int(block),
        "env": copy.deepcopy(env_log),
        "reward": float(reward),
        "terminal": bool(terminal)
    }
    log["record"].append(event)

check_every = 50

start_time = datetime.datetime.now()

for ep in range(NUM_EPISODES):
    episode = START_EP + ep
    start_block = 0
    check_block = check_every
    print(f'===== EPISODE {episode} =====')

    state, reward, terminal = env.reset()

    print('== PLACING BLOCKS ==')
    while not terminal:
        env_actions = todo()

        # print('= ENV makes STEP =')
        next_state, reward, terminal = env.step(env_actions)

        log_everything(episode, env.block_seq_index, env.log, reward, terminal)

        # print(env.block_seq_index)
        if ((env.block_seq_index+1) >= check_block) or terminal:
            print(f'{datetime.datetime.now()} TRAINING: Block {env.block_seq_index}, Reward = {reward:.2f}, Percent complete = {env.perc_complete*100:.2f}%')
            # print('!!!!SAVE!!!!')
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