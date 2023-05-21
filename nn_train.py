import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from robot_env import RobotEnv1

# Define the DQN agent
class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.model = self.build_model()

    def build_model(self):
        # Neural network for deep Q-learning
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_shape=(None, 2), activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        return model

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state):
        # Update the Q-table
        target = reward + 0.99 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the CartPole environment
env = RobotEnv1(mapPath="map.txt")
action_size = 4

# Create the DQN agent
agent = DQNAgent(action_size)

# Iterate over episodes
for episode in range(1000):
    state = env.getState()
    action = agent.act(state)
    next_state, reward, = env.move(action)
    agent.train(state, action, reward, next_state)
    state = next_state

    print(f'Episode: {episode + 1},  Exploration Rate: {agent.epsilon}')

# Close the environment
