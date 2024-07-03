A world model computes a representation of the real world (i.e., via encoder, hidden representation) and predicts the next state of the world.

[[LeCun World Model Definition]]
Given:
 - an observation x(t)
 - a previous estimate of the state of the world s(t)
 - an action proposal a(t)
 - a latent variable proposal z(t)

A world model computes: [[world-model-represents-and-predicts]]
 - representation: h(t) = Enc( x(t) )
 - prediction: s(t+1) = Pred( h(t), s(t), z(t), a(t) )
- Where
 - Enc() is an encoder (a trainable deterministic function, e.g. a neural net)
 - Pred() is a hidden state predictor (also a trainable deterministic function).
 - the latent variable z(t) represents the unknown information that would allow us to predict exactly what happens. It must be sampled from a distribution or or varied over a set. It parameterizes the set (or distribution) of plausible predictions.