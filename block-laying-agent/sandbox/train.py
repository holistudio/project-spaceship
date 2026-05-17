def main():
    episodes = 5

    # env = BlockEnvironment()

    # teacher_agent = TeacherAgent()
    # block_agent = PPOBlockAgent()

    # agent_list = [teacher_agent, block_agent]
    # agent_list = [teacher_agent, 'random']
    # agent_list = [user, block_agent]

    for ep in range(episodes):
        print(f"{"#"*5} EPISODE {ep+1:,} {"#"*5}")
        # idx = 0
        # env.reset()
        # for a in env.agent_iter():
            # agent = agent_list[idx]
            # observation, reward, termination, truncation, info = env.last()
            
            # if termination or truncation:
                # actions = None
            # else:
                # mask = observation["actions_mask"]
                # if agent == 'random':
                    # actions = env.actions_space.sample(mask)
                # else:
                    # actions = agent.step(observation)
            
            # env.step(actions)

            # if idx % 2 == 0:
                # TODO: revisit this later in terms of 
                # associating the reward correctly to 
                # block_agent's PREVIOUS (NOT LATEST) obs/actions
                # agent.experience(observation, actions, reward)
                
            # idx = (idx+1)%2
    # env.close()
    pass


if __name__ == "__main__":
    main()