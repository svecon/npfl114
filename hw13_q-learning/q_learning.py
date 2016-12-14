#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import environment_discrete
import numpy as np

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Taxi-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=3000, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=1000, type=int, help="Maximum number of steps.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
    parser.add_argument("--alpha_decay", default=0.000001, type=float, help="Learning rate decay rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Epsilon.")
    parser.add_argument("--epsilon_decay", default=0.00001, type=float, help="Epsilon decay rate.")
    parser.add_argument("--gamma", default=0.9999, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = environment_discrete.EnvironmentDiscrete(args.env)

    # Create Q and other variables
    Q = np.zeros([env.states, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha
    gamma = args.gamma
    episode_rewards, episode_lengths = [], []

    for episode in range(args.episodes):
        # Perform episode
        state = env.reset()
        total_reward = 0
        for t in range(args.max_steps):
            if args.render_each and episode > 0 and episode % args.render_each == 0:
                env.render()

            # Compute action using epsilon-greedy policy
            if np.random.random() < epsilon:
                action = np.random.randint(0, env.actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update Q
            Q[state][action] = Q[state][action] + alpha*( reward + ( gamma*np.max(Q[next_state]) ) - Q[state][action] )

            state = next_state
            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(t)
        if len(episode_rewards) % 10 == 0:
            print("Episode {}, mean 100-episode reward {}, mean 100-episode length {}, epsilon {}, alpha {}.".format(
                episode + 1, np.mean(episode_rewards[-100:]), np.mean(episode_lengths[-100:]), epsilon, alpha))

        if args.epsilon_decay:
            epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_decay)]))
        if args.alpha_decay:
            alpha = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_decay)]))
