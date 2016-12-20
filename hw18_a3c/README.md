# HW-18 A3C (3pts, due Jan 09)

Note that this task is experimental and may not be easily solvable!

Try implementing Asynchronous Advantage Actor Critic algorithm from Asynchronous Methods for Deep Reinforcement Learning paper. You can start with the labs11/a3c-skeleton.py module.

You will get the points is you can show minor improvement, reaching average reward of at 100 on CartPole-v1. Do not hesitate to send the solution even if it is unstable.

Note that the network frequently diverges – in addition to gradient clipping (present in the skeleton), you could use exponential learning rate decay, or some entropy regularization term (see the paper).
