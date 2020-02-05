
import numpy as np
import gym
import pickle

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import kerpy


# Create models
def CreateModel(env):
    n = env.unwrapped.spec.id
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16, name=n+'_input'))
    model.add(Activation('relu', name='activation1'))
    model.add(Dense(16, name="dense2"))
    model.add(Activation('relu', name='activation2'))
    model.add(Dense(16, name='dense3'))
    model.add(Activation('relu', name='activation3'))
    model.add(Dense(nb_actions, name=n+'_output'))
    model.add(Activation('linear', name='activation4'))
    model.name = env.unwrapped.spec.id
    return model


# Compile and configure models for each problem
def Compile(model, env):
    if len(env.action_space.shape):
        actions = env.action_space.shape[0]
    else:
        actions = env.action_space.n
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=actions,
                   memory=memory, nb_steps_warmup=100,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn
