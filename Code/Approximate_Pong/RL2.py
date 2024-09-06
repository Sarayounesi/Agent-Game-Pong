import gymnasium as gym
import numpy as np
import os.path
# import cv2
import random
import math

gamma = 1
alpha = 0.2
last_epsilon = 0.05
epsilon = 0.05
Limit = 1000 * 1000
ACTIONS = [0, 2, 3, 4, 5]
WEIGHTS = {"W1": 1, "W2": 1, "W3": 1, "W4": 1, "W5": 1}
x1 = 0
y1 = 0





def SaveQToFile():
    with open("WInfos.txt", 'w') as updateWeight:
        updateWeight.write('%f %f %f \n' %
                           (WEIGHTS["W1"], WEIGHTS["W2"], WEIGHTS["W3"]))


def ReadQFromFile():
    if (os.path.exists("WInfos.txt")):
        with open("WInfos.txt", 'r') as updateWeight:
            line = updateWeight.readline()
            WEIGHTS["W1"], WEIGHTS["W2"], WEIGHTS["W3"] = line.split()
            return float(WEIGHTS["W1"]), float(WEIGHTS["W2"]), float(WEIGHTS["W3"])
    return 0, 0, 0


def Ball(state):
    for i in range(34, 194):  # amoodi
        for j in range(0, 160):  # ofoghi
            if state[i][j][0] == 236 and state[i][j][1] == 236 and state[i][j][2] == 236:
                return i, j
    return 0, 0
# Ball


def NextStateOfAgent(state, action):
    dy = 0
    if action == 3:
        dy = -10
    if action == 2:
        dy = 10
    for i in range(34, 194):  # amoodi
        for j in range(140, 144):  # ofoghi
            if state[i][j][0] == 92 and state[i][j][1] == 186 and state[i][j][2] == 92:
                return i + dy + 8, j

    return 0, 0
# PlatePos


def PlateEnemyPos(state):
    for i in range(34, 194):
        for j in range(16, 19):
            if state[i][j][0] == 213 and state[i][j][1] == 130 and state[i][j][2] == 74:
                return i, j


def BallPlateDistanceToAgent(state, action):
    x1, y1 = Ball(state)
    x2, y2 = NextStateOfAgent(state, action)
    return abs(x2 - x1)/100, abs(y2 - y1) / 100


def Feature3(state):
    counter = 0
    for i in range(34, 193):
        for j in range(138, 141):
            if state[i][j][0] == 236 and state[i][j][1] == 236 and state[i][j][2] == 236 and state[i][j+1][0] == 92 and state[i][j+1][1] == 186 and state[i][j+1][2] == 92:
                counter += 1
    return counter
# fiercount


def Feature4(state):
    counter = 0
    for i in range(34, 194):
        for j in range(0, 16):
            if state[i][j][0] == 236 and state[i][j][1] == 236 and state[i][j][2] == 236:
                counter += 1
    return counter


def Feature5(state):
    x1, y1 = Ball(state)
    x2, y2 = PlateEnemyPos(state)
    return abs(x2-x1)/100, abs(y2-y1)/100
# BallPlateDistanceToEnemy


def updateWeight(state, action):
    dx, dy = BallPlateDistanceToAgent(state, action)
    return WEIGHTS["W1"] * dx + WEIGHTS["W2"] * dy - WEIGHTS["W3"]*Feature3(state)


env = gym.make("ALE/Pong", render_mode="human")
state, info = env.reset(seed=5)
WEIGHTS["W1"], WEIGHTS["W2"], WEIGHTS["W3"] = ReadQFromFile()

for i in range(10000):
    action = None
    optimizedValue = math.inf
    for optimizedAction in ACTIONS:
        value = updateWeight(state, optimizedAction)
        if optimizedValue > value:
            optimizedValue = value
            action = optimizedAction

    if action is None or random.random() <= epsilon:
        action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    futureReward = 0
    for a in ACTIONS:
        print(a)
        futureReward = max(futureReward, updateWeight(next_state, a))
    difference = (reward + gamma * futureReward) - updateWeight(state, action)
    dx, dy = BallPlateDistanceToAgent(state, action)
    WEIGHTS["W1"] = round(WEIGHTS["W1"] + alpha * difference * dx, 6)
    WEIGHTS["W2"] = round(WEIGHTS["W2"] + alpha * difference * dy, 6)
    WEIGHTS["W3"] = round(WEIGHTS["W3"] + alpha *
                          difference * Feature3(state), 6)
    print(WEIGHTS["W1"],WEIGHTS["W2"],WEIGHTS["W3"],WEIGHTS["W4"] )
    state = next_state
    if terminated or truncated:
        next_state, info = env.reset()
SaveQToFile()

env.close()

# env = gym.make("Pong-v0", render_mode='rgb_array')
# env = gym.wrappers.RecordVideo(env, 'video', step_trigger = lambda x: x <= <number of iterations > , name_prefix='output', video_length= < number of iterations > )
