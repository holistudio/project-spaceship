**Link:** https://doi.org/10.1016/j.neunet.2022.03.037

**Authors**: Yutaka Matsuo, [[LeCun, Yann]], Maneesh Sahani, Doina Precup, [[Silver, David]],
Masashi Sugiyama, Eiji Uchibe, Jun Morimoto

## Look Into
 - 

## Problem

"many robots were needed to be involved to collect large-scale data from a real environment to acquire reasonable-level policies. Unlike image recognition tasks, collecting data for learning robot controllers is quite time-consuming and sometimes impossible since the robot needs to interact with its physical environment. "

"physical simulation to virtually train a policy and applying the acquired one to a real system would be a promising method to cope with the difficulty of collecting data in the real environment. Although this sim-to-real approach has been successfully implemented, for example, in a hierarchical RL framework (Morimoto & Doya, 2001) or with using domain randomization (Akkaya et al., 2019), so far, there are only limited applications."

Deep learning, RL, and "other approaches such as Bayesian inference (Ghahramani, 2015) and symbolic reasoning methods (Russell & Peter Norvig, 2020) are also important."

"In this review, we summarize talks and discussions in the “Deep Learning and Reinforcement Learning” session of the symposium, International Symposium on Artificial Intelligence and Brain Science (AIBS2020)."

**World model for perception, control and language** - Yutaka Matsuo

"a world model as a simulator in our brain."

"We can learn the world model using deep generative models."

 **Self-supervised learning** - Yann LeCun

"Self-Supervised learning is basically learning to fill in the blanks (video clip, text and so on)."
### Specific Challenges

- "collecting data for learning robot controllers is quite time-consuming and sometimes impossible since the robot needs to interact with its physical environment."
- "individual theories are not sufficient to achieve a comprehensive understanding of our brain. "

**World model for perception, control and language** - Yutaka Matsuo

 - "If the amount of information related to the missing modality is large, it might result in a collapsed representation. This is what we call the missing modality problem."
 
 **Self-supervised learning** - Yann LeCun
 
 - "Babies seem to learn basic concepts about the world in the first few months of life. What type of learning is taking place in the brain when babies perform this kind of learning? Trying to figure out the process is the biggest obstacle to making real progress in AI."
 - "how to represent uncertainty/multi-modality in the prediction"
 - "We need to do a process of developing the prediction, coming up with the best action, executing the action, and then repeating the process. And this may have to be done for multiple drawings of latent variables, which may be very costly."

## Related Works

 - ...
 - ...

## Approach

### World model for perception, control and language
Yutaka Matsuo

"Encoders for the respective modalities were prepared. Then we apply learning to them to approximate the original Variational Auto Encoder (VAE). After training, we can use each trained encoder for proper inference from a single modality. This method, Joint Multimodal VAE (JMVAE), can obtain the joint representation and well perform bidirectional generation because it explicitly learns to recover a missing modality from the observed modality. "

"Behavior Regularized Offline Reinforcement Learning (BREMEN) (Matsushima, Furuta, Matsuo, Nachum, & Gu, 2020). It not only performs better than the state-of-the-art approaches on existing benchmarks, but it can also optimize a policy offline effectively using only a tenth or a twentieth of the data necessary for earlier methods. BREMEN learns a dynamics model, which can be regarded as a world model, from the offline dataset. It interacts with the learned model. The algorithm is based on Dyna-style model-based RL, learning an ensemble of dynamics models in conjunction with a policy using imaginary rollouts."

### Self-supervised learning
Yann LeCun

Two uses of self-supervised learning:
"learning hierarchical representations of the world...used in supervised learning or RL afterward."
"learning predictive (forward) models of the world...used for model-predictive control or model-based RL."

"how to represent uncertainty/multi-modality in the prediction. For this, Energy-Based Model was proposed (LeCun, Chopra, Hadsell, Marc’Aurelio, & Huang, 2006). There are two types of methods for training Energy-Based Model, contrastive methods and regularized/architectural methods."

"Contrasting methods have been extremely successful in recent years, particularly for applications in natural language processing. In the process of predicting missing words in the text, the system will learn good representations of texts that can be used in subsequent tasks"
"system is trained to learn a common representation between two identical networks. There has been a considerable success with techniques like PIRL (Misra & van der Maaten, 2019), MoCo (He, Fan, Wu, S., & Girshick, 2019), SimCLR (Chen, Kornblith, Norouzi, & Hinton, 2020). But the problem with contrastive learning is that it does not scale very well because it takes a lot of computation to train the system."

"In regularized/architectural methods, a latent variable model is constrained to be sparse. Therefore, its information content is limited. It limits the volume of space that can take energy."

**Learning Predictive Forward Models**

"By gradient descent, we can find a sequence of actions that will minimize an objective in optimal control that is called the adjoint state method."

"Instead of doing a gradient descent with respect to action every time, training a policy network of neural net to predict the right action that will minimize cost."

## Key Findings

 - ...
 - ...
 - ...

## Takeaways

### World model for perception, control and language

"lower part (sensory-motor system), which comprises world models and a controller, deals with real-world patterns and takes actions based on that...The upper part (symbol system) deals with language...Especially, a world model is triggered by language; it is used as a simulator. A deep generative model conditioned on input sentences is used for that purpose. We can imagine many things such as flying cars and mountain-high giants by using language. Thereby, we call it a mental canvas. " (See Figure 1)

### Self-supervised learning

"Cost function indicates instantaneous cost of the state of the world. Critic would be a trainable function which is going to estimate or predict in advance what the ultimate cost of an outcome is going to be. Actor is going to either run this policy network or in case that skill has not been completely acquired yet, it will basically infer a sequence of action and optimize the cost through optimization. And there is a need for a perception module that estimates world state" (See Figure 2)

Figure 2 shows some AI's offline learning most likely. But what happens when the AI is operating and learning online? Actor runs the policy network on the world model internally, but I think Critic ultimately helps decide what the AI will do in the real-world externally. Then perception module observes what happens in the real world and the entire loop runs again.

 - ...
 - ...
 - ..
