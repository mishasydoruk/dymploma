import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Activation, Flatten, Dense
from robot_env import RobotEnv1
import random
from collections import deque



# Define the Deep Q-Network (DQN) model


# Define the Agent class
class DQNAgent:
    def __init__(self, state_size, action_size, usePretrained=False):
        self.optimizer = Adam(learning_rate=0.01)
        self.path = "checkpoint.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.expirience_replay = deque(maxlen=2000)
        self.gamma = 0.6
        self.epsilon = 0.1
        self.model = create_model(state_size, action_size)
        if usePretrained:
            self.restore()
        self.target_model = create_model(state_size, action_size)


    def store(self, state, action, reward, next_state):
        self.expirience_replay.append((state, action, reward, next_state))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([i for i in range(self.action_size)])
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state)[0][0]
            print(q_values)
            action = np.argmax(q_values)
        return action

    def train_model(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state in minibatch:

            target = self.model.predict(state)

            t = self.target_model.predict(next_state)
            target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)

    def save(self):
        self.model.save_weights(self.path)

    def restore(self):
        self.model.load_weights(self.path)

# Define the main function for training the agent
def main():
    env = RobotEnv1("map.txt", sellSize=26)
    state_size = (1, 130, 130)
    action_size = 4

    agent = DQNAgent(state_size, action_size, False)
    state = env.getState()
    total_reward = 0
    for episode in range(10000):

        action = agent.get_action(state)
        next_state, reward = env.move(action)
        agent.train_model(state, action, reward, next_state)
        total_reward += reward
        state = next_state

        print("Episode:", episode, "Total Reward:", total_reward)

        if episode % 4 == 0:
            agent.target_model.set_weights(agent.model.get_weights())

        if episode % 100 == 0:
            agent.save()




if __name__ == "__main__":
    main()