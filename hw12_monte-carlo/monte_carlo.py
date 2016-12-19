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
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.5, type=float, help="Epsilon.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Epsilon decay rate.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = environment_discrete.EnvironmentDiscrete(args.env)

    # Create Q, C and other variables
    Q = np.zeros([env.states, env.actions])
    C = np.zeros([env.states, env.actions])
    epsilon = args.epsilon
    episode_rewards, episode_lengths = [], []

    for episode in range(args.episodes):
        # Perform episode
        state = env.reset()
        states, actions, rewards, total_reward = [], [], [], 0
        for t in range(args.max_steps):
            if args.render_each and episode > 0 and episode % args.render_each == 0:
                env.render()

            if np.random.random() < epsilon:
                action = np.random.randint(0, env.actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # Update Q and C
        for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
            g = sum([ x*(args.gamma**j) for j,x in enumerate(rewards[i:]) ]) # this could take O(n) instead of O(n^2) if the outer for loop was reversed
            C[state][action] += 1
            Q[state][action] = Q[state][action] + (1.0 / C[state][action]) * (g - Q[state][action]) # Equivalent function (gradient-like)
            # Q[state][action] = Q[state][action]*(C[state][action]-1)/C[state][action] + (g / C[state][action]) # Equivalent function (more intuitive?)

        episode_rewards.append(total_reward)
        episode_lengths.append(t)
        if len(episode_rewards) % 10 == 0:
            print("Episode {}, mean 100-episode reward {}, mean 100-episode length {}, epsilon {}.".format(
                episode + 1, np.mean(episode_rewards[-100:]), np.mean(episode_lengths[-100:]), epsilon))

        if args.epsilon_final:
            epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
