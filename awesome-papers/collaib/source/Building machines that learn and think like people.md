---
tags:
  - "#AI"
  - "#reasoning"
  - "#RL"
  - generative_model
  - model_based_RL
  - neural_net
  - CNN
---

**Link:** https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/building-machines-that-learn-and-think-like-people/A9535B1D745A0377E16C590E14B94993

**Authors**: [[Lake, Brenden]] [[Ullman, Tomer]] [[Tenenbaum, Joshua]] [[Gershman, Samuel]]

## Look Into
 - 

## Notes

### 1. Introduction

Speech recognition
- Hidden Markov Models (HMMs) were once the state of the art in 1980s but later dominated by deep learning (Hinton)

Complex control now dominated by DRL like DQN

This article:
- reviews cognitive science, developmental psychology, and AI
- posits ideas from cognitive science
- identifies gaps between deep learning and those cognitive science ideas
towards developing AI that learns and thinks like a person

There are two different computational approaches to intelligence:
- Pattern recognition - prediction is the goal, and discovering features that have high-value states is the primary means
- Model building - models of the world are primary to understanding the world and explaining its nature

"pattern recognition, even if it is not the core of intelligence, can nonetheless support model building" by making "essential inferences more computationally efficient"

"At a more fundamental level, any computational model of learning must ultimately be grounded in the brain's biological neural networks."
"As long as natural intelligence remains the best example of intelligence, we believe that the project of reverse engineering the human solutions to difficult computational problems will continue to inform and advance AI."

Future neural nets can have:
- intuitive physics
- theory of mind
- causal reasoning
- structure
- inductive biases built in or learning from previous experience

There are examples of AI that does NOT draw inspiration from aspects of human cognition. (footnote: Wright brothers stopped observing birds and started looking at wind tunnels and aerodynamics)

"Other human cognitive abilities remain difficult to understand computationally, including creativity, common sense, and general-purpose reasoning."

Most exciting are new forms of probabilistic machine learning. See:
- Ghahramani, 2015
- Lloyd et al. 2014
- Grosse et al. 2012
- Gelman et al. 2015

Core ingredients of human-like learning:
- early development cognitive abilities - regardless of tabula rasa or innate in nature, likely foundational to all other tasks that need to be done for more complex problems
	- intuitive physics - stuff is solid and tends to fall or follow certain trajectories, helps to discount implausible trajectories in predicting the future
	- intuitive psychology - understanding that people have goals, beliefs, agency
- causal model building
- compositionality
- learning-to-learn
- fast runtime in putting learned models into action

Here's the thing with RL:
- model based RL is great in incorporating models of the world for better actions to maximize reward, but very slow at runtime, making it bad for real-time control
- model-free RL is fast but not necessarily best for optimizing controls to maximize the same reward function

"We review evidence that humans combine model-based and model-free learning algorithms both competitively and cooperatively and that these interactions are supervised by metacognitive processes."

### 2. Cognitive and neural inspiration in artificial intelligence

