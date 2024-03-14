import numpy as np
import CNN_agent as Agent
import training_env as Env

NUM_EPISODES = 10

env = Env.BlockTrainingEnvironment()
agent = Agent.CNNAgent(grid_sizes=env.grid_sizes, 
                       num_orient=env.num_orient, 
                       block_info_size=env.block_info_size, 
                       block_types=env.block_types)

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

        state = next_state
    
    print(f'==END EPISODE {ep}==')
    print()
    print()

    agent.update_policy(ep+1)