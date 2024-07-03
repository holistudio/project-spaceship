
**Link:** https://arxiv.org/abs/2211.03136

**Authors**: ...

## Look Into
 - "Recently, multi-agent deep reinforcement learning (MARL) has been employed to synthesize spatial configuration (Veloso and Krishnamurti, 2021). In their framework, each agent represents a specific spatial partition exploring the environment and communicating with each other to achieve certain objectives related to satisfying some geometrical and topological constraints. However, their approach relies on the assumption that the outline is not restricted to a certain shape and size.

## Problem
  
"Alexander believes that every design process is involved finding a fitness between two intangible components: the form and its context. The context defines the design problem and the form offers a solution for that problem, meaning that the designer’s goal is to create a form that fits various requirements of its context. The design process includes an initial form that is born in the context, and altering the form in a series of steps until reaching the desired design solution."

"The analogy between the RL problem and the design process can be touchable if we consider the RL agent as an artificial designer, the environment as the context, and the learned behavior as the form."

"The real challenge in the design process is that we want to find **harmony between two intangible components: a form that has not been designed yet, and a context that cannot be fully described.** Similarly, in RL we need an optimal policy to maximize the reward but we have not yet learned the policy and often we do not access the environment’s model that generates the reward."  

"Spatial layout planning concerns shaping, arranging, and dimensioning spacial elements to satisfy **geometrical, topological, and performance constraints** while following certain objectives"  

Spatial layout planning "highly affects the building efficiency and end-user comfortability and determines the top-level spatial structure of a building from the early stages of the design process"  

"Due to innumerable alternatives in configuration, finding feasible layout design solutions falls in the NP-complete category, for which finding an optimal solution might be impossible"

"the RL algorithm we use will **not suffer from simulation to real-world performance degradation** as there is no need to run the agent outside of the simulator."

### Specific Challenges

**Lingering Problems with RL**
- generalizability across different contexts / design scenarios.
- transitioning from exploration to exploitation is not straightforward.
- "RL methods usually suffer from generalization abilities that make it difficult to transfer knowledge between different design scenarios. Moreover, for any design scenario, there could be multiple solutions that might prevent the training process from efficiently transitioning from exploration to exploitation as the agent could switch between different solutions during the training process."

## Related Works

 - ...
 - ...

## Approach

**Dataset:** ...

Conventional plan partitioning action sequences via rectangular dissection really sucks (Figure 2)

1. Actions: laser-walls - "a hard part including two segments making the base of a wall, and a soft part including two segments that extend the base wall to create a complete wall...hen a base wall is placed in the plan (the thick segments in Figure 3.b), it emits the light from both segments to extend them (the thin segments in Figure 3.b) until the lights hit either a surface with a higher infiltration rate, the hard part of a wall, or the outline...A laser-wall also has a few specifications. First, the soft part of a laser-wall has a certain infiltration rate. The more the infiltration rate of the new wall is the stronger the new light is in cutting the other existing lights. Second, the light cannot infiltrate into the base walls (the hard part of the walls) and the outline." So basically a wall could cut into an existing wall with some probability (i.e., if the new wall has a higher "infiltration rate") rather than all walls are entirely hard and set in stone throughout the design process. Figure 3

2. State space: "Divide the plan into a 2D grid of discrete cells out of which the state can be defined in two ways:"

- image representation where the state is the RGB image of the plan OR

- feature vector representation which includes the coordinates of each wall, the distance between five important points of each wall and the other walls, the distance between five important points of each wall and the four main corners of the outline.

3. "The planning terminates when all walls are being placed. As the single-agent environment includes static walls, we name the single-agent environment as one-shot planning."

## Key Findings

 - ...
 - ...
 - ...

## Takeaways

 - ...
 - ...
 - ..

