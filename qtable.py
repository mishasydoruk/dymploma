import random


class QTable:
    def __init__(self, actionSize=4, filename="train2.txt"):
        self.actionSize = actionSize
        self.table = self.read(filename)
        self.filename = filename

    def update(self, state, action, new_value):
        if self.table.get(state) is None:
            self.table[state] = [0 for i in range(self.actionSize)]

        self.table[state][action] = new_value

    def get(self, state, action):
        actions = self.table.get(state)

        if actions is None:
            self.table[state] = [0 for i in range(self.actionSize)]
            return random.choice(range(self.actionSize))
        result = actions[action]
        return result

    def getBestAction(self, state):
        arr = self.table.get(state)
        if arr is None:
            self.table[state] = [0 for i in range(self.actionSize)]
            return random.choice(range(self.actionSize))
        return arr.index(max(arr))

    def getMaxValue(self, state):
        arr = self.table.get(state)
        if arr is None:
            return 0
        return max(arr)

    def read(self, filename):
        res = dict()
        with open(filename) as file:
            for line in file:
                state = line.split(",")
                res[state[0]] = [float(i) for i in state[1: self.actionSize + 2]]
        return res

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        keysList = list(self.table.keys())
        file = open(filename, "w")
        for key in keysList:
            file.write(key+","+",".join([str(i) for i in self.table.get(key)])+"\n")
