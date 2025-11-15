class SudoAgent(object):
    def __init__(self):
        self.latest = {
            'action': "",
            'value': "",
            'logp': ""
        }
        self.exp_buffer # experience buffer / replay memory
        pass

    def record(self, observation, action, next_observation, reward):
        """
        record the observation => action => next_observation, reward
        transition, regardless of whether the action
        came from this agent or some other agent
        """
        # self.exp_buffer.store(...)
        pass
    
    def step(self, observation, mask=None):
        """

        """

        # compute action, value, logp

        # store action, value, logp
        # self.latest['action'] = ...
        # self.latest['value'] = ...
        # self.latest['logp'] = ...

        # return action to the environment for its next step
        return action
    
    def update(self):
        """
        update agent policy using experience buffer / replay memory
        """
        pass
    
