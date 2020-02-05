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


def ActorCritic(env):
    n = env.unwrapped.spec.id + '_actor'
    nb_actions = env.action_space.shape[0]
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16, name=n+'_dense1', activation='relu'))
    #actor.add(Activation('relu', name='activation1'))
    actor.add(Dense(16, name='dense2', activation='relu'))
    #actor.add(Activation('relu', name='activation2'))
    actor.add(Dense(16, name='dense3', activation='relu'))
    #actor.add(Activation('relu', name='activation3'))
    actor.add(Dense(nb_actions, name=n+'_output', activation='linear'))
    #actor.add(Activation('linear', name='activation4'))
    actor.name = n

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape,
                              name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    critic.name = env.unwrapped.spec.id + '_critic'
    return actor, critic, action_input, observation_input


def ActorCriticCompile(env, actor, critic, action_input, observation_input):
    nb_actions = env.action_space.shape[0]
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                              theta=.15,
                                              mu=0.,
                                              sigma=.3)

    dqn_acrobot = DDPGAgent(nb_actions=nb_actions,
                            actor=actor,
                            critic=critic,
                            critic_action_input=action_input,
                            memory=memory,
                            nb_steps_warmup_critic=100,
                            nb_steps_warmup_actor=100,
                            random_process=random_process,
                            gamma=.99,
                            target_model_update=1e-3)

    dqn_acrobot.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    return dqn_acrobot
