import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.utils import plot_model

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
        model.add(Conv2D(64,3,3, border_mode='same',input_shape = (84,84,1),activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(32,activation = 'relu'))
        model.add(Dense(self.actions, activation = 'linear'))
        model.compile(loss = 'mean_squared_error',optimizer = Adam(lr = self.learningRate))
        model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True,
                   show_layer_names=True)
        #print("model")
        #model.predict(np.zeros((1,84,84,1)))
        return model

    def act(self,state):
        #if np.random.rand() <= self.epsilon:
            #return random.randrange(self.actions)
        actionList = self.model.predict(np.array([state]))
        #print ("action list")
        #print (actionList.shape)
        return np.argmax(actionList[0])

    def addToMemory(self,state,action,reward,nextState,done):
        self.memory.append((state,action,reward,nextState,done))

    def replay(self, batchSize):
        batch = random.sample(self.memory,batchSize)
        for state,action,reward,nextState,done in batch:
            target = reward
            #print ("Next State")
            #print (nextState.shape)
            if not done:
               target = (reward + self.gamma * np.amax(self.model.
                                        predict(np.array([nextState]))))

            result = self.model.predict(np.array([state]))
            result[0][action] = target
            self.model.fit(np.array([state]),result,epochs=1,verbose=0)
        if self.epsilon > self.epsilonMini:
            self.epsilon *= self.epsilonDecay
