import random
from robot_env import RobotEnv
from qtable import QTable

alpha = 0.7
gamma = 0.9
epsilon = 0

env = RobotEnv("map.txt", chance=20)
qtable = QTable()

reward = 0
state = env.getCurrentState()
i = 1
for i in range(10000000):

    if random.uniform(0, 1) < epsilon:
        action = env.getRandomAction()
    else:
        action = qtable.getBestAction(state)

    next_state, reward = env.move(action)

    old_value = qtable.get(state, action)
    next_max = qtable.getMaxValue(next_state)

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    qtable.update(state, action, new_value)

    state = next_state

    if i % 10000 == 0:
        print(i)

qtable.write("train.txt")
