The design space can be defined by a set of parameters and metrics, both of which require special considerations.

[[Beyond Heuristics - A Novel Design Space Model for Generative Space Planning in Architecture]]
 - A design space model is defined as a a finite set of input parameters + evaluation of each solution based on pre-defined metrics.
 - Design space should be defined such that it has sufficient bias, variance, complexity, and continuity but not too much. This can be checked with a sparse sampling of parameters before optimization
 - Their use of "design model" may be more accurately called the design parameter space
 - there is "little discussion of how the design space model can be evaluated by designers before the optimization is run."
 - the quality of the design space model can be evaluated in terms of:
	 - Bias and Variance tradeoff - what is the scope of the design space model?\
		 - High bias - too simple doesn't capture enough variety of designs
		 - High variance - too many possible designs and more likely to include a lot of invalid designs.
	 - Complexity vs continuity - what is the internal structure of the design space model?\
	 - Complexity vs Continuity
	 - Too little complexity - why use a computer in the first place?
	 - Too little continuity - algorithms have a hard time figuring out trends in the results and how to best modify the design
	- "complex enough to allow unexpected solutions, yet continuous enough to be searchable."
