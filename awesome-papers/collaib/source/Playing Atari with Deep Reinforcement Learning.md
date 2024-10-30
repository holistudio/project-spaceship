
## Problem

"Most successful RL applications that operate on these domains have relied on hand-crafted features"

"the performance of such systems heavily relies on the quality of the feature representation"

"The delay between actions and resulting rewards, which can be thousands of timesteps long, seems particularly daunting when compared to the direct association between inputs and targets found in supervised learning"

CNNs, restricted Boltzman machines, RNNs can be seen as methods for feature extraction from raw sensory data - is there still a need for hand-crafted features?

a scalar reward signal that is frequently sparse, noisy and delayed
 - The delay is a particular issue relative to supervised learning drawing immediate associations
 - "sequences of highly correlated states"

## Solution

CNN with input from raw video data from Atari 2600 games (210x160 RGB at 60Hz)
 - the agent only observes images of the current screen, the task is partially observed and many emulator states are perceptually aliased, i.e. it is impossible to fully understand the current situation from only the current screen

"The network is trained with a variant of the Q-learning"

"To alleviate the problems of correlated data and non-stationary distributions, we use an experience replay mechanism, which randomly samples previous transitions, and thereby smooths the training distribution over many past behaviors"