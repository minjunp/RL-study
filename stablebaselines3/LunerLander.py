# Type of RL algorithms
# https://stable-baselines3.readthedocs.io/en/master/guide/algos.html

import gym
from stable_baselines3 import A2C, PPO

env = gym.make("LunarLander-v2")
env.reset()

# Multi-layer perceptron
model = A2C("MlpPolicy", env, verbose=1)
# Train for 10000 steps
# The total number of samples (env steps) to train on
model.learn(total_timesteps=50000)

episodes = 5
for ep in range(episodes):
# for step in range(500):
    # Reset environment in every episode
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # obs, reward, done, info = env.step(env.action_space.sample())
        # print(reward)
env.close()