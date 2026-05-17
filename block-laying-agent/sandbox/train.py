from environment import BlockEnvironment

def main():
    episodes = 5

    max_rounds = 3
    env = BlockEnvironment(max_timesteps=max_rounds*2)

    # teacher_agent = TeacherAgent()
    # block_agent = PPOBlockAgent()

    agent_list = ['random', 'random']
    # agent_list = [teacher_agent, block_agent]
    # agent_list = [teacher_agent, 'random']
    # agent_list = [user, block_agent]

    for ep in range(episodes):
        print(f"\n\n{"#"*5} EPISODE {ep+1:,} {"#"*5}\n")
        
        idx = 0 # for tracking whose turn it is on agent_list

        env.reset()
        while not (env.termination or env.truncation):
            print(env.timestep, env.current_player)
            agent = agent_list[idx]
            observation, reward, termination, truncation = env.last()
            
            if termination or truncation:
                actions = None
            else:
                mask = observation["action_mask"]
                if agent == 'random':
                    action = env.sample_action(mask)
                    actions = {"new_blocks": action}
                else:
                    actions = agent.step(observation)
            
            env.step()
            # env.step(actions)

            # if env.current_player == 'agent':
                # TODO: revisit this later in terms of 
                # associating the reward correctly to 
                # block_agent's PREVIOUS (NOT LATEST) obs/actions
                # agent.experience(observation, actions, reward)
                
            idx = (idx+1)%2
    # env.close()
    pass


if __name__ == "__main__":
    main()