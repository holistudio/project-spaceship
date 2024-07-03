
**Link:** https://papers.cumincad.org/data/works/att/acadia17_436.pdf

**Authors**: [[Nagy, Danil]], [[Villagi, Lorenzo]], [[Zhao, Dale]], [[Benjamin, David]]

## Look Into

 - Calixto and Celani (2015) review of metaheuristics in space planning
 - Flack and Ross (2011), who propose a space planning model based on the subdivision and merging of contiguous plan areas.

## Problem

heuristics in design "are not guaranteed to produce the best overall solution. In the context of design, they can also lead to "design fixation," where "designers limit their creative output because of an overreliance on features of preexisting designs" (Youmans and Arciszewski, 2014)" [[design-heuristics-limit-creativity]]

"metaheuristics are stochastic algorithms that can find optimal solutions to complex problems by iteratively sampling possible solutions, evaluating them based on specified performance factors, and using this information to derive better and better designs" [[meta-heuristics-help-generative-design]]

Advantages of generative design [[generative-design-advantages]]
- wider range of possible solutions considered
- break free of heuristics found in traditional processes
- novel yet high performing solutions as determined by simulations evaluating designs

### Specific Challenges

 - But you really need a well-defined design space model - a finite set of input parameters + evaluation of each solution based on pre-defined metrics. [[design-space-model-needs-to-be-well-defined]]
 - There is "little discussion of how the design space model can be evaluated by designers before the optimization is run."
 - Quality of the design space model can be evaluated in terms of:
	 - Bias and Variance tradeoff - what is the scope of the design space model?
	 - Complexity vs continuity - what is the internal structure of the design space model?

## Related Works

 - ...
 - ...

## Approach

**Dataset:** ...

Design space model inspired by urban morphology [[design-principles-apply-across-scale]]

### Design Space Definition

- Constraints: no-go zones of egress/restrooms etc, overall boundary, entrances to the floor
- Avenues subdivide the space into a collection of smaller parcels/cells (parameter set 1)
- Three avenues starting points fixed at entrances, end points can move around the boundary edges (3 parameters)
- Merge parcels into larger regions to meet program requirements (parameter set 2)
- 11 major expo programs that branch off immediately from the avenues - 1 point along the avenue line (seed point) and a vector that stops when the smaller parcel/cell areas add up to a program's requirements: "A neighbor is chosen for merging with the starting cell, such that the resulting area minimally meets the area requirements of the given program. If all neighbors fall short of meeting the area requirement, the largest neighbor is chosen for merging.
- "To reduce the dimensionality of the design space, we chose not to individually parameterize the location of the F&B programs. Instead, these programs are always located at the ends of the primary and secondary avenues, where they hit the exterior walls of the expo hall." 

### Design Space Metrics

Main goal of the exhibit space: "all exhibitor areas are evenly activated without creating undesirable congestion in any particular area" [[evaluate-design-based-on-occupant-experience]]

- ""buzz," measures the spatial distribution of high traffic areas in the plan (Figure 4a)...large amount of foot traffic, but distribute this foot traffic evenly throughout the plan."
- "the average foot traffic around each exhibitor booth (Figure 4b). Higher values of exposure represent plans where a large percentage of booths have a high level of traffic surrounding them."
- "static graph-based simulation method, which is fully described in a related paper" 

### Design Space Evaluation
[[understanding-state-space]] before running/optimizing models can be helpful

Bias vs variance [[design-space-model-needs-to-be-well-defined]]
- High bias - too simple doesn't capture enough variety of designs
- High variance - too many possible designs and more likely to include a lot of invalid designs.

Complexity vs Continuity
- Too little complexity - why use a computer in the first place?
- Too little continuity - algorithms have a hard time figuring out trends in the results and how to best modify the design
- "complex enough to allow unexpected solutions, yet continuous enough to be searchable."

Their use of "design model" may be more accurately called the design parameter space

"discretize each \[of the three avenue end points] input into 16 steps, which yields 16 x 16 x 16 = 4,096 designs" [[grid-size-affects-state-space]]

"To smooth out the effect of the remaining 22 input parameters we also tested each design five times using random values for the other parameters and averaged the values of the output metrics." [[random-sampling-addresses-high-dimensionality]]

## Key Findings

 - Continuity vs complexity: The plots show us that the transition between high and low scores is gradual in certain zones and more abrupt in others. In general, the landscapes show a degree of overall continuity while revealing local complexities between the input parameters and the output metric scores (Figure 6 inset). The design space also seems to be more continuous along the parameter represented along the y axis than the one along the x axis.
 - "We can also learn about the scope of the design space by studying the form of resulting plans at its boundaries. As stated earlier, the merging process guarantees that all options within the space are valid designs. Thus we are not in danger of creating a model with too much variance. However, we can also see that the initial subdivision process leads to a wide variation of plan designs. Thus, our model is not overly biased towards particular solutions such as orthogonal layouts."
	 - weird how this is a qualitative conclusion and not something evaluated numerically. 
 - Design space evaluation reveals key design principles for achieving design objectives: "For example, we can see that the high performing region of the design space in the top left x-y quadrants is produced whenever both avenues miss the central obstacle in the exhibit hall."

## Takeaways

- [[understanding-state-space]] before running/optimizing models can be helpful
- Design space should be defined such that it has sufficient bias, variance, complexity, and continuity but not too much. This can be checked with a sparse sampling of parameters before optimization
- Once it's clear that the design space is sufficiently complex, metaheuristics like MOGA can be used with confidence.
- In the face of high dimensional state spaces, random sampling may suffice rather than sampling every point in the high dimensional space [[random-sampling-addresses-high-dimensionality]]






