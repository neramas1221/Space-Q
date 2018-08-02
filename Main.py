import random
import gym
import numpy as np
import cv2
from tensorflow.python.client import device_lib
from Agent import Agent
print(device_lib.list_local_devices())

def updateImage(state):
    state = cv2.cvtColor(cv2.resize(state,(84,110)),cv2.COLOR_BGR2GRAY)
    #remove the first 26 layers
    state = state[26:110,:]
    _,state = cv2.threshold(state, 1,255,cv2.THRESH_BINARY)
    #print(state.shape)
    state = np.reshape(state,(84,84,1))
    return state#.reshape(1,state.shape[0],state.shape[1],state.shape[2])

Episodes = 1000
Trials = 500
env = gym.make('SpaceInvaders-v0')

states = env.observation_space.shape

actions = env.action_space.n
#print(actions)
agent = Agent(states,actions)
done = False
batchSize = 32

for i in range (Episodes):
    state = env.reset()
    state = updateImage(state)
    #print ("State shape")
    #print (state.shape)


    while not done:
        #print (state.shape)
        env.render()
        action = agent.act(state)
        nextState,reward,done,inf   = env.step(action)
        nextState = updateImage(nextState)
        #print("next state")
        #print(len(state))
        #reward = reward if not done else 0.01
        #nextState = np.reshape(nextState,[1,states])
        agent.addToMemory(state,action,reward,nextState,done)
        state = nextState
        if done:
            print("episode {}/{}, score : {}, e: {:.2}".format(i,
                      Episodes,reward,agent.epsilon))
            break
        if len(agent.memory) > batchSize:
                agent.replay(batchSize)
    done = False
