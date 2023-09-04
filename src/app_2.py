import tensorflow as tf
import gym
import random
import numpy as np

# Hyperparameters
EPOCHS = 1000
STEPS = 200
GAMMA = 0.9
EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995

# Create environment
env = gym.make('CartPole-v1')

# Q-Network
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

state = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
target = tf.placeholder(shape=[None], dtype=tf.float32)
action = tf.placeholder(shape=[None], dtype=tf.int32)

W1 = tf.Variable(tf.random_normal([state_size, 10]))
B1 = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.random_normal([10, action_size]))
B2 = tf.Variable(tf.zeros([action_size]))

h1 = tf.nn.relu(tf.matmul(state, W1)+B1)
logits = tf.matmul(h1, W2)+B2

loss = tf.losses.huber_loss(target, logits)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Training
for e in range(EPOCHS):

    state = env.reset()
    for t in range(STEPS):

        action_values = sess.run(logits, feed_dict={state: [state]})
        action = np.argmax(action_values[0]) if random.random() < EPSILON else random.choice(range(action_size))

        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1

        # Training
        target_value = reward
        sess.run(optimizer, feed_dict={
            state: [state],
            target: [target_value]
        })

        state = next_state

    # Decay epsilon
    EPSILON = max(MIN_EPSILON, EPSILON*EPSILON_DECAY)
