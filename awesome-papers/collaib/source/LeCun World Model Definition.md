**Source:** https://www.linkedin.com/posts/yann-lecun_lots-of-confusion-about-what-a-world-model-activity-7165738293223931904-vdgR

**Authors:** [[LeCun, Yann]]

**Quotes:**
Lots of confusion about what a world model is. Here is my definition:

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
 - the latent variable z(t) represents the unknown information that would allow us to predict exactly what happens. It must be sampled from a distribution or or varied over a set. It parameterizes the set (or distribution) of plausible predictions.

The trick is to train the entire thing from observation triplets (x(t), a(t), x(t+1)) while preventing the Encoder from collapsing to a trivial solution on which it ignores the input. [[encoder-collapse-is-a-problem]]

Auto-regressive generative models (such as LLMs) are a simplified special case in which
 1. the Encoder is the identity function: h(t) = x(t),
 2. the state is a window of past inputs
 3. there is no action variable a(t)
 4. x(t) is discrete
 5. the Predictor computes a distribution over outcomes for x(t+1) and uses latent z(t) to select one value from that distribution.

The equations reduce to:

s(t) = \[x(t), x(t-1),...,x(t-k)]

x(t+1) = Pred( s(t), z(t) )

There is no collapse issue in that case.