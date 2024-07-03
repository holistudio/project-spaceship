
[[LeCun World Model Definition]]
"The trick is to train the entire thing from observation triplets (x(t), a(t), x(t+1)) while preventing the Encoder from collapsing to a trivial solution on which it ignores the input.

Auto-regressive generative models (such as LLMs) are a simplified special case in which the Encoder is the identity function: h(t) = x(t),... There is no collapse issue in that case."