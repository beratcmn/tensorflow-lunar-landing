import random
import gym
import tensorflow as tf

env = gym.make("CartPole-v1", render_mode="human")


episodes = 10

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0, 1])
        n_state, reward, done, truncated, info = env.step(action)
        score += reward

    print(f"Episode: {episode} Score: {score}")
