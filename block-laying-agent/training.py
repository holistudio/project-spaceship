import os
import json
import copy
import datetime

from environment import BlockEnvironment
from teacher_agent import TeacherAgent

NUM_EPISODES = 2
DIR = os.path.join('results', 'teacher')

START_EP = 0

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

env = BlockEnvironment()


terminal = False
a_i = 0

start_time = datetime.datetime.now()

episode = START_EP
start_block = 0
check_block = check_every

print(f'===== EPISODE {episode} =====')

env.reset()
agent0 = TeacherAgent(env.target_tensor)
agent1 = TeacherAgent(env.target_tensor)
agents = [agent0, agent1]

print('== PLACING BLOCKS ==')
while not terminal:
    agent = agents[a_i]

    observation, reward, terminal, info = env.last()

    if terminal:
        next_observation = None
    else:
        action_mask = info['action_mask']
        action = agent.step(observation, action_mask)
        next_observation = env.step(action)

    # agent1.record(observation, action, next_observation, reward)

    if a_i < len(agents)-1:
        a_i += 1
    else:
        a_i = 0

    log_everything(episode, env.block_ix-1, env.env_log, reward, terminal)
    if ((env.block_ix+1) >= check_block) or terminal:
        print(f'{datetime.datetime.now()} TRAINING: Block {env.block_ix}, Reward = {reward:.2f}, Percent complete = {env.perc_complete*100:.2f}%')
        log_file = os.path.join(DIR,f'episode_{episode}_blocks_{start_block}-{env.block_ix}_log.json')
        start_block = env.block_ix+1
        check_block += check_every

        with open(log_file, 'w') as f:
            json.dump(log, f)

        # Clear log
        log = {
            "record":[]
        }


# NUM_ATTEMPTS = 10

# while not terminal:
    
    # observation, reward, terminal, info = env.last()

    # for attempt in range(NUM_ATTEMPTS)
        # attempt_action = agent.step(observation)
        # valid = env.check_action(attempt_action)
        # if valid: break

    # if valid:
        # action = attempt_action
    # else:
        # maybe let the environment add another valid block 
        # but then have the agent record that action and 
        # next state as its experience

    # env.step(action)

    # if terminal:
        # next_observation = None
    # else:
        # next_observation = env.step(action)

    # agent.record(observation, action, next_observation, reward)