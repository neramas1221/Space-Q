import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent:


    def __init__ (self, states, actions)
        self.states = states
        self.actions = actions
        self.memory = deque(maxlen = 1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilonMini = 0.01
        self.epsilonDecay = 0.995
        self.learningRate = 0.001
        self.model = self.makeModel()

    def makeModel(self)

        model = Sequential()
        model.add(Dense(24,input_dim = self.states,activation = 'relu'))
        model.add(Dense(24,activation = 'relu'))
        model.add(Dense(self.actions, activation = 'linear'))
        model.compile(loss = 'mse',optimizer = Adam(lr = self.learningRate))
        return model
