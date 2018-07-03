import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent:


    def __init__ (self, states, actions):
        self.states = states
        self.actions = actions
        self.memory = deque(maxlen = 1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilonMini = 0.01
        self.epsilonDecay = 0.995
        self.learningRate = 0.001
        self.model = self.makeModel()

    def makeModel(self):
        model = Sequential()
        model.add(Dense(24,input_dim = self.states,activation = 'relu'))
        model.add(Dense(24,activation = 'relu'))
        model.add(Dense(self.actions, activation = 'linear'))
        model.compile(loss = 'mse',optimizer = Adam(lr = self.learningRate))
        return model
    
    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.actions)
        actionList = self.model.predict(state)
        return np.argmax(actionList[0])
    
    def addToMemory(self,state,action,reward,nextState,done):
        self.memory.append(state,action,reward,nextState,done)
        
    def replay(self, batchSize):
        batch = random.sample(self.memory,batchSize)
        for state,action,reward,nextState,done in batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.
                                predict(nextState)[0]))
            result = self.model.predict(state)
            result[0][action] = target
            self.model.fit(state,target,epochs=1,verbose=0)
        if self.epsilon > self.epsilonMini:
            self.epsilon *= self.epsilonDecay
        