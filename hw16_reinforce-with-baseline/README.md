# HW-16 reinforce-with-baseline (2pts, due Jan 09)

Implement REINFORCE algorithm with value function as a baseline, representing both a policy and a value function using (independent) neural networks with a hidden layer. Start with the labs11/reinforce_with_baseline-skeleton.py module.

You should be able to reach average reward of 490 on CartPole-v1 environment (using 500 steps) and -90 on Acrobot-v1 environment.

To observe the effect of the baseline, try comparing your solution to basic `reinforce` using batch of size 1.
