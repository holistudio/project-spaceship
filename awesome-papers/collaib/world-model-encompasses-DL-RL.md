A world model is one possible way to connect concepts of RL for selecting actions with other supervised and unsupervised learning methods for encoding and predicting:

[[LeCun World Model Definition]]

Given:
 - an observation x(t)
 - a previous estimate of the state of the world s(t)
 - an action proposal a(t) [[world-model-encompasses-DL-RL]]
 - a latent variable proposal z(t)

A world model computes: [[world-model-represents-and-predicts]]
 - representation: h(t) = Enc( x(t) )
 - prediction: s(t+1) = Pred( h(t), s(t), z(t), a(t) )

Where
 - Enc() is an encoder (a trainable deterministic function, e.g. a neural net)
 - Pred() is a hidden state predictor (also a trainable deterministic function).
 - the latent variable z(t) represents the unknown information that would allow us to predict exactly what happens. It must be sampled from a distribution or or varied over a set. It parameterizes the set (or distribution) of plausible predictions. [[latent-variables-help-fill-data-gaps]]