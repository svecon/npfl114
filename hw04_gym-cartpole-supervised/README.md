# HW-04 gym_cartpole_supervised (3pts, due Nov 14)

Solve the CartPole-v1 (https://gym.openai.com/envs/CartPole-v1) environment from the OpenAI Gym using supervised learning. Very small amount of training data is available in the labs04/gym-cartpole-data.txt file, each line containing one observation (four space separated floats) and a corresponding action (the last space separated integer).

The solution to this task should be a model which passes evaluation on random inputs. This evaluation is performed by running the labs04/gym-cartpole-evaluate.py model_file command. (You can also pass –render argument to render the evaluations interactively.) In order to pass, you should achieve an average reward of at least 475 on 100 episodes.

In order to save the model, look at the labs04/gym-cartpole-save.py, which saves a model performing random guesses.
