import random
from robot_env import RobotEnv
from qtable import QTable
import matplotlib.pyplot as plt

alpha = 0.4
gamma = 0.4
epsilon = 0.9

vals = []

for i in range(100000):
    env = RobotEnv("map2.txt", chance=0)
    qtable = QTable(filename="train2_2.txt")
    reward = 0
    state = env.getCurrentState()
    epsilon *= 0.97
    done = False
    i = 0
    while not done:
        i += 1
        env.render()
        if random.uniform(0, 1) < epsilon:
            action = env.getRandomAction()
        else:
            action = qtable.getBestAction(state)
        new_state, reward, done = env.move(action)
        if done:
            reward += 10
        old_value = qtable.get(state, action)
        next_max = qtable.getMaxValue(new_state)

        print(state, new_state, action, reward, sep=" ")
        print()

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        qtable.update(state, action, new_value)
        state = new_state
    qtable.write("train2_2.txt")
    print(i)
    vals.append(i)
    plt.clf()
    plt.plot(vals)
    plt.draw()
    plt.pause(0.0001)
