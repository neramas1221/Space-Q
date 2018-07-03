import random
import gym
import numpy as np
from tensorflow.python.client import device_lib
from Agent import Agent
print(device_lib.list_local_devices())

Episodes = 100
Trials = 1000
env = gym.make('SpaceInvaders-v0')

states = env.observation_space.sample()# shape
states = np.array(states).flatten()
print states.shape
actions = env.action_space.n
agent = Agent(states,actions)
done = False
batchSize = 1

for i in range (Episodes):
    state = env.reset().flatten()
    #state = np.reshape(state,[1, states])
    for p in range (Trials):
        #print state.shape
        env.render()
        action = agent.act(state)
        nextState,reward,done,_   = env.step(action)
        nextState = np.array(nextState).flatten()
        reward = reward if not done else +10
        
        #nextState = np.reshape(nextState,[1,states])
        agent.addToMemory(state,action,reward,nextState,done)
        state = nextState
        if done:
            print("episode {}/{}, score : {}, e: {:.2}".format(i,
                      Episodes,p,agent.epsilon))
            break
        if len(agent.memory) > batchSize:
                agent.replay(batchSize)
    