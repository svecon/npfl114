# HW-17 reinforce-with-baseline-pixels (3pts, due Jan 09)

Note that this task is experimental and may not be easily solvable!

Modify the solution of `reinforce_with_baseline` to use pixel inputs. Start with the labs11/reinforce_with_baseline_pixels-skeleton.py module.

You will get the points is you can show any improvement at all, reaching for example average reward of 50 on CartPole-v1.

Note that according to papers, it could take hours for the network to converge. Also note that you probably has to use some kind of epsilon-greedy policy (otherwise the policy network usually converges too fast to a wrong solution; in some papers [for example in Asynchronous Methods for Deep Reinforcement Learning] entropy regularization term is used instead).
