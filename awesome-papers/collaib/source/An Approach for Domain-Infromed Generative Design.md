
**Link:** ...

**Authors**: [[Keundeok Park]], Semiha Ergan

## Questions
- What's the difference between adjacency and connectivity and integration?
- If room sizes are randomly decided, what ensures that the generated designs have a good amount of floor area for each room?
- Are door placements on shared walls also in the DRL's action space?
- What was the neural network used by the DQN?
	-  Huber Loss?
- "empirically to balance with the previous criterion" meaning?
- What software was used for the integration calculations?
- After each integration value is calculated for each cell, each room can then be scored by summing up the integration values of all its cells?
- How does Figure 7 show random agent's 64% floor plans had higher integration for living rooms than master bedrooms?
## Look Into
 - [Dueling Networks for DQN](https://arxiv.org/pdf/1511.06581)

## Problem

 Architects and engineers need to consider a wide and diverse set of design alternatives during the design phase of building projects
 
 At the same time, they have to judge the design alternatives according to complex sets of criteria including costs, code compliance, and quality of life to future occupants.
 
 This process of design exploration and evaluation can take longer and prove more costly with more complex projects.

### Specific Challenges

 - Architectural floor layout is a particular sub problem in the design phase the requires architects to consider space adjacency, connectivity, and utilization
	 - Space adjacency - Which spaces are best arranged closer together (i.e., bedroom and bathroom)?
	 - Space connectivity - When an occupant is standing in a particular space, how visible should other spaces be, so that there is a sense of connection between these spaces?
	 - Integration - How open/public or closed off/private is the space, based on features like distance from the building entrance and placement of openings (i.e., doors) between spaces. 
	 - Utilization - Do the floor areas for each space meet the client's requirements (i.e., program requirements)? Do the dimensions of the space (length, width, height) and ratios of those dimensions suit the needs of the space (i.e., corridors might be best if they are long and narrow, but living rooms should not)
 - Architects generally rely on heuristics or rules of thumb to visually inspect floor plans rather than rely on quantitative metrics and computational methods. This limits the design exploration significantly.
 - Generative design methods have been proposed to help with design exploration for architectural floor layout but run into problems when it comes to generating layouts that meet the evaluation criteria above.
	 - Generative adversarial networks (GANs) in particular have emerged as a data-driven method that can take hundreds of existing floor layout images and generate new ones.
	 - But the quality of GANs generated designs depends more on the existing dataset of images. Garbage in-garbage out.
	 - Moreover GANs models can contains hundreds of hidden parameters that make it difficult to control and fine tune internally. 
	 - When evaluating GANs floor plans in a previous studies, most were not found to meet the criteria of space adjacency, connectivity, and utilization.

Deep reinforcement learning (DRL) has a potential to be easier to control for design exploration of floor plan layouts:
- DRL has been shown to do well in games with increasingly complex sets of objectives and dynamic environments.
- Design phase in building projects have analogous conditions where requirements have a tendency to change over the course of the design phase.
- DRL does not require a large existing dataset like GANs.
- DRL does require an explicit reward or objective function. This means that it is possible to define the objectives of the DRL algorithm using the domain knowledge of architects, and easier to control towards space adjacency, connectivity, and utilization requirements.
## Related Works

Automating floor layout generation has had previous research

 - Heuristic approaches: shape grammar, mathematical programming, and genetic algorithms. While these approaches can define the criteria or objective function to match domain knowledge, they also tend to over-prescribe or over defining a procedure to coming up with layouts, and end up exploring a limited set of design alternatives.
 - Data driven: Using large datasets and having deep learning algorithms learn patterns of floor layouts themselves, these data driven methods do not require a pre-defined procedure for layouts and can therefore explore a wider range of alternatives.
	 - GANs use two networks, generator and discriminator. While the generator proposes a floor plan layout, the discriminator scores the generator on how much that proposal resembles the real layouts in the dataset. Over time, the generator will propose more realistic layouts. This approach has a few limitations (see above)
	 - DRL approaches have also been explored for floor layouts and incorporated adjacency and connectivity into their reward function.

## Approach

2-bedroom / 2-bathroom apartment unit: living room, kitchen, master bedroom, master bathroom, second bedroom, and the second bathroom (5 spaces)

DQN DRL deciding on discrete actions for room length and width, location, and door placement in a 25x25 grid 

1. DRL specifies room size and location
2. DRL specifies where doors are placed on shared walls between rooms
3. Reward function calculated for adjacency, intersection over union (iou), and integration
	1. Living room is the largest room in layout (`+1`)
	2. Adjacency required in this predefined set: `C = {(living, bed1), (bed1, bath1), (living, bed2), (living, bath2)}` Given layout, add up which spaces are correctly adjacent (`+r_adj`where `r_adj` empirically balances with the previous criterion)
	3. IOU = - (Area of intersection between two spaces) / (Combined Union Area of two spaces). The greater the overlap, the greater the area of intersection, the more negative this reward, the greater the penalty fed back to DRL.
	4. Integration: For a given grid cell in a room, measures the number of visual steps between that grid cell and all other grid cells. The higher this measurement, the more integrated that grid cell is to the the rest of the room AND the other rooms that are connected to it. After each integration value is calculated for each cell, each room can then be scored by summing up the integration values of all its cells.
4. Once all five rooms are placed, the episode ends

Compare DRL with a random agent that always randomly specifies room size, room location, and doors between rooms.
1. First 2000 episodes by DRL were random so that the DRL can learn from a wide variety of possible layouts and associate them to rewards
2. Then trained DRL for 20000 episodes and compared reward with random action.


## Key Findings

 - Layouts better than random after training for 20000 episodes, given the running average of last 100 episodes shown in Figure 5
 - Occasionally DRL still proposed bathrooms placed awkwardly in the middle of the bedroom. Refining the reward function can fix this issue or adding hard constraints to its possible actions.
 - Analysis of the last 100 floor layouts by the DRL show that its living room tend to have greater integration than master bedroom. This indicates the DRL can produce layouts with appropriate levels of privacy in each space, ensuring a better quality of life for occupants.
 - 95% floor plans had higher integration for living rooms than master bedrooms, while in the random group, 64% floor plans had higher integration for living rooms than master bedrooms.

## Takeaways

 - ...
 - ...
 - ..





