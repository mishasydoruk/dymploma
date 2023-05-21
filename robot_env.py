import copy
import random

import numpy as np
import cv2

import matplotlib.pyplot as plt

class RobotEnv:
    symb = {
        "0": "block",
        "1": "dirt",
        "3": "robot",
        "2": "clear",
        "block": "0",
        "dirt": "1",
        "robot": "3",
        "clear": "2"
    }

    colors = {
        "block": (0, 0, 0),
        "clear": (255, 255, 255),
        "dirt": (200, 200, 200),
        "robot": (0, 0, 255)
    }

    actions = ["R", "L", "U", "D"]

    robotSym = "3"
    stateSize = 4

    def __init__(self, mapPath="map.txt", sellSize=20, chance=1):
        self.map = self.readMap(mapPath)
        self.chance = chance
        self.robotPosition = self.index_2d(self.map, self.robotSym)
        self.scaleSize = sellSize
        self.vals = []

    def move(self, direction):
        newPoint = copy.copy(self.robotPosition)

        direction = self.actions[direction]

        if direction == "L":
            newPoint[1] -= 1

        if direction == "R":
            newPoint[1] += 1

        if direction == "U":
            newPoint[0] -= 1

        if direction == "D":
            newPoint[0] += 1

        newPointReward = 0

        if self.map[newPoint[0]][newPoint[1]] == self.symb["block"]:
            newPointReward = -1

        if self.map[newPoint[0]][newPoint[1]] == self.symb["clear"]:
            newPointReward = -1

        if self.map[newPoint[0]][newPoint[1]] == self.symb["dirt"]:
            newPointReward = 1

        self.dirt()
        if self.map[newPoint[0]][newPoint[1]] != self.symb["block"]:
            self.paint(newPoint[0], newPoint[1], "robot")
            self.paint(self.robotPosition[0], self.robotPosition[1], "clear")
            self.robotPosition = newPoint
            self.render()
            #self.renderState()

        #return self.getCurrentState(), newPointReward, sum(row.count(self.symb["dirt"]) for row in self.map) == 0
        return self.getCurrentState(), newPointReward

    def dirt(self):
        coef = len(self.map)
        coef2=coef/2
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if (random.choice(range(5000)) < abs(coef2-i)*abs(coef2-j)*self.chance/100) and self.symb[self.map[i][j]] == "clear":
                    self.paint(i, j, "dirt")

    def paint(self, x, y, action):
        self.map[x][y] = self.symb[action]

    def readMap(self, filePath, sellSize=1):
        res = []
        with open(filePath) as file:
            for line in file:
                res.append([i for i in line.replace("\n", "")])
        return self.scale(res, sellSize)

    def index_2d(self, myList, v):
        for i, x in enumerate(myList):
            if v in x:
                return [i, x.index(v)]

    def scale(self, map, times=20):
        res = []
        for line in map:
            for n in range(times):
                res.append([i for i in self.repeat(line, times)])
        return res

    def repeat(self, arr, times):
        return self.flatten([[i for j in range(times)] for i in arr])

    def flatten(self, arr):
        return [item for sublist in arr for item in sublist]

    def getCoverage(self):
        return (sum(row.count(self.symb["clear"]) for row in self.map) + 1) / (
                    len(self.map) * len(self.map[0]) - sum(row.count(self.symb["block"]) for row in self.map))

    def getRandomAction(self):
        return random.choice(range(len(self.actions)))

    def getState(self):

        begX = max(0, self.robotPosition[0]-self.stateSize)
        endX = min(len(self.map), self.robotPosition[0]+self.stateSize)
        begY = max(0, self.robotPosition[1]-self.stateSize)
        endY = min(len(self.map), self.robotPosition[1]+self.stateSize)

        return [self.map[i][begY:endY] for i in range(begX,endX)]

    def getCurrentState(self):
        return str(self.robotPosition[0])+" "+str(self.robotPosition[1])+" "+"".join(self.flatten(self.getState()))
        #return str(self.robotPosition[0])+" "+str(self.robotPosition[1])

    def renderState(self):
        state = self.getState()
        scaledImg = self.scale(state, self.scaleSize*4)
        sizex = len(scaledImg[0])
        sizey = len(scaledImg)
        displayImage = np.array([[self.colors[self.symb[scaledImg[i][j]]] for j in range(sizex)] for i in range(sizey)],
                                dtype=np.uint8)
        cv2.imshow("State ", displayImage)

    def render(self):
        scaledImg = self.scale(self.map, self.scaleSize)
        sizex = len(scaledImg[0])
        sizey = len(scaledImg)
        displayImage = np.array([[self.colors[self.symb[scaledImg[i][j]]] for j in range(sizex)] for i in range(sizey)],
                                dtype=np.uint8)
        cv2.imshow("Game ", displayImage)
        self.vals.append(self.getCoverage())
        print("Coverage = " + str(self.getCoverage()))
        plt.clf()
        plt.plot(self.vals)
        plt.draw()
        plt.pause(0.0001)
        cv2.waitKey(1)

class RobotEnv1:
    symb = {
        0.0: "block",
        1.0: "dirt",
        3.0: "robot",
        2.0: "clear",
        "block": 0.0,
        "dirt": 1.0,
        "robot": 3.0,
        "clear": 2.0
    }

    colors = {
        "block": (0, 0, 0),
        "clear": (255, 255, 255),
        "dirt": (200, 200, 200),
        "robot": (0, 0, 255)
    }

    actions = ["R", "L", "U", "D"]

    robotSym = 3
    stateSize = 2

    def __init__(self, mapPath="map.txt", sellSize=20, chance=1):
        self.map = self.readMap(mapPath)
        self.chance = chance
        self.robotPosition = self.index_2d(self.map, self.robotSym)
        self.scaleSize = sellSize
        self.vals = []

    def move(self, direction):
        newPoint = copy.copy(self.robotPosition)

        direction = self.actions[direction]

        if direction == "L":
            newPoint[1] -= 1

        if direction == "R":
            newPoint[1] += 1

        if direction == "U":
            newPoint[0] -= 1

        if direction == "D":
            newPoint[0] += 1

        newPointReward = 0

        if self.map[newPoint[0]][newPoint[1]] == self.symb["block"]:
            newPointReward = -1

        if self.map[newPoint[0]][newPoint[1]] == self.symb["clear"]:
            newPointReward = 0

        if self.map[newPoint[0]][newPoint[1]] == self.symb["dirt"]:
            newPointReward = self.getCoverage()*10

        self.dirt()
        if self.map[newPoint[0]][newPoint[1]] != self.symb["block"]:
            self.paint(newPoint[0], newPoint[1], "robot")
            self.paint(self.robotPosition[0], self.robotPosition[1], "clear")
            self.robotPosition = newPoint
            self.render()
            #self.renderState()

        return self.getState(), newPointReward

    def dirt(self):
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if (random.choice(range(5000)) < self.chance) and self.symb[self.map[i][j]] == "clear":
                    self.paint(i, j, "dirt")

    def paint(self, x, y, action):
        self.map[x][y] = self.symb[action]

    def readMap(self, filePath, sellSize=1):
        res = []
        with open(filePath) as file:
            for line in file:
                res.append([float(i) for i in line.replace("\n", "")])
        return self.scale(res, sellSize)

    def index_2d(self, myList, v):
        for i, x in enumerate(myList):
            if v in x:
                return [i, x.index(v)]

    def scale(self, map, times=20):
        res = []
        for line in map:
            for n in range(times):
                res.append([i for i in self.repeat(line, times)])
        return res

    def repeat(self, arr, times):
        return self.flatten([[i for j in range(times)] for i in arr])

    def flatten(self, arr):
        return [item for sublist in arr for item in sublist]

    def getCoverage(self):
        return (sum(row.count(self.symb["clear"]) for row in self.map) + 1) / (
                    len(self.map) * len(self.map[0]) - sum(row.count(self.symb["block"]) for row in self.map))

    def getRandomAction(self):
        return random.choice(range(len(self.actions)))

    def getState(self):

        # begX = max(0, self.robotPosition[0]-self.stateSize)
        # endX = min(len(self.map), self.robotPosition[0]+self.stateSize)
        # begY = max(0, self.robotPosition[1]-self.stateSize)
        # endY = min(len(self.map), self.robotPosition[1]+self.stateSize)

        #return self.scale(self.map, 5)
        return [self.robotPosition[0], self.robotPosition[1]]

    def getCurrentState(self):
        #return str(self.robotPosition[0])+" "+str(self.robotPosition[1])+" "+"".join(self.flatten(self.getState()))
        return str(self.robotPosition[0])+" "+str(self.robotPosition[1])

    def renderState(self):
        state = self.getState()
        scaledImg = self.scale(state, self.scaleSize*4)
        sizex = len(scaledImg[0])
        sizey = len(scaledImg)
        displayImage = np.array([[self.colors[self.symb[scaledImg[i][j]]] for j in range(sizex)] for i in range(sizey)],
                                dtype=np.uint8)
        cv2.imshow("State ", displayImage)

    def render(self):
        scaledImg = self.scale(self.map, self.scaleSize)
        sizex = len(scaledImg[0])
        sizey = len(scaledImg)
        displayImage = np.array([[self.colors[self.symb[scaledImg[i][j]]] for j in range(sizex)] for i in range(sizey)],
                                dtype=np.uint8)
        cv2.imshow("Game ", displayImage)
        self.vals.append(self.getCoverage())
        print("Coverage = " + str(self.getCoverage()))
        plt.clf()
        plt.plot(self.vals)
        plt.draw()
        plt.pause(0.0001)
        cv2.waitKey(1)