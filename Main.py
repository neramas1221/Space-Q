import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
