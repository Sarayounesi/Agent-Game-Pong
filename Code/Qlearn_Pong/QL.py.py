import gymnasium as gym
import numpy as np
import cv2
import random

Q = {}
gamma = 1
alpha = 0.2
last_epsilon = 0.07
epsilon = 0.1
ACTIONS = [0, 2, 3, 4, 5]


def Func1(image):
    res = ""
    for i in range(len(image)):
        for j in range(len(image[i])):
            res = res + str(image[i][j])
    return res


def Func2(image):
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 1
    return Func1(image_data)


def SaveQToFile():
    with open("Q.txt", 'w') as f:
        for key, value in Q.items():
            state, action = key
            f.write('%s %d %f\n' % (state, action, value))


def ReadQFromFile():
    with open("Q.txt", 'r') as f:
        for line in f.readlines():
            state, action, value = line.split()
            Q[(state, int(action))] = float(value)


env = gym.make("ALE/Pong-v5")

state, data = env.reset(seed=42)
state = Func2(state)
ReadQFromFile()


for _ in range(10000000000):
    action = None
    r = 0
    for a in ACTIONS:
        if (state, a) in Q and r < Q[(state, a)]:
            r = Q[(state, a)]
            action = a
    if action is None or random.random() <= epsilon:
        action = env.action_space.sample()
    N, R, T, Tc, data = env.step(action)
    N = Func2(N)

    if not (state, action) in Q:
        Q[(state, action)] = 0.0

    if Q[((state, action))] != 0.0:
        print("action for  train  ", action,
              "  State , Action", Q[((state, action))])

    Next_Next = 0
    for a in ACTIONS:
        if (N, action) in Q:
            Next_Next = max(Next_Next, Q[(N, action)])

    Q[(state, action)] = (1 - alpha) * Q[(state, action)] + \
        alpha * (R + gamma * Next_Next)

    if epsilon > last_epsilon:
        epsilon -= 0.001
    state = N
    if T or Tc:
        N, data = env.reset()
SaveQToFile
env.close()
