Understanding the state space before running models can be helpful

[[Beyond Heuristics - A Novel Design Space Model for Generative Space Planning in Architecture]]
- Design space should be defined such that it has sufficient bias, variance, complexity, and continuity but not too much. This can be checked with a sparse sampling of parameters before optimization[[design-space-model-needs-to-be-well-defined]]
- Bias vs variance 
	- High bias - too simple doesn't capture enough variety of designs
	- High variance - too many possible designs and more likely to include a lot of invalid designs.
- Complexity vs Continuity
	- Too little complexity - why use a computer in the first place?
	- Too little continuity - algorithms have a hard time figuring out trends in the results and how to best modify the design
	- "complex enough to allow unexpected solutions, yet continuous enough to be searchable."
-  "We can also learn about the scope of the design space by studying the form of resulting plans at its boundaries. As stated earlier, the merging process guarantees that all options within the space are valid designs. Thus we are not in danger of creating a model with too much variance. However, we can also see that the initial subdivision process leads to a wide variation of plan designs. Thus, our model is not overly biased towards particular solutions such as orthogonal layouts."
- Design space evaluation reveals key design principles for achieving design objectives: "For example, we can see that the high performing region of the design space in the top left x-y quadrants is produced whenever both avenues miss the central obstacle in the exhibit hall."