from dqn import *
from actor_critic import *
import time
import gym


class Env_class():
    def __init__(self, model_name):
        self.env = None
        self.model = None
        self.critic = None
        self.config = {'verbose': 2, 'max_epi_step': None}
        self.history = {'train': None, 'test': None}
        self.compiled = None
        self.recompile_func = None
        self.info = {'train_time': None, 'test_time': None}
        split = model_name.split('-')
        self.fullname = model_name
        self.version = split[1]
        self.model_name = split[0]
        names = {'cartpole': self.create_cartpole,
                 'acrobot': self.create_acrobot,
                 'pendulum': self.create_pendulum}
        names[self.model_name]()

    def create_cartpole(self):
        self.env = gym.make('CartPole-'+self.version)
        self.model = CreateModel(self.env)
        self.compiled = Compile(self.model, self.env)
        self.recompile_func = Compile

    def create_acrobot(self):
        self.env = gym.make('Acrobot-'+self.version)
        self.model = CreateModel(self.env)
        self.compiled = Compile(self.model, self.env)
        self.recompile_func = Compile

    def create_pendulum(self):
        self.config['verbose'] = 1
        self.config['max_epi_step'] = 200
        self.env = gym.make('Pendulum-'+self.version)
        out = ActorCritic(self.env)
        self.model = out[0]
        self.critic = out[1]
        self.action_input = out[2]
        self.observation_input = out[3]
        self.compiled = ActorCriticCompile(self.env,
                                           self.model,
                                           self.critic,
                                           self.action_input,
                                           self.observation_input)

        self.recompile_func = self.compiled.load_weights

    def train(self, steps=50000, enable_visual=False):
        assert self.compiled is not None, "Model has nopt been properly setup"
        t1 = time.time()
        h = self.compiled.fit(self.env,
                              nb_steps=steps,
                              visualize=enable_visual,
                              verbose=self.config['verbose'],
                              nb_max_episode_steps=self.config['max_epi_step'])
        self.info['train_time'] = time.time() - t1
        self.history['train'] = h.history
        self.env.close()
        return h.history

    def test(self, episodes=5, max_steps=300, enable_visual=False, verbose=1):
        t1 = time.time()
        h = self.compiled.test(self.env,
                               nb_episodes=episodes,
                               visualize=enable_visual,
                               nb_max_episode_steps=max_steps,
                               verbose=verbose)
        self.info['test_time'] = time.time() - t1
        self.history['test'] = h.history
        self.env.close()
        return h.history

    def load_model(self, model_path, critic_path=None, layers_to_freeze=None):
        """ """
        self.model.load_weights(model_path, by_name=True)
        if self.critic:
            # Cosas critic
            #if critic_path != 'main':
            #    self.critic.load_weights(critic_path)
            self.compiled.actor = self.model
            self.compiled.critic = self.critic
            self.compiled.update_target_models_hard()
            print("Modelos cargandos correctamente")
        else:
            # Cosas normales
            self.compiled = self.recompile_func(self.model, self.env)
            print(f'Model {model_path} loaded')

        # Frozen Layers
        if layers_to_freeze is not None:
            if layers_to_freeze == 'all':
                for i, l in enumerate(self.model.layers):
                    self.model.layers[i].trainable = False
            else:
                if isinstance(layers_to_freeze[0], str):
                    for name in layers_to_freeze:
                        self.model.get_layer(name=name).trainable = False
                if isinstance(layers_to_freeze[0], int):
                    for index in layers_to_freeze:
                        self.model.get_layer(index=index).trainable = False

    def save_model(self, path='OUTPUTS/', name='default', pickle_out=True):
        if self.critic is not None:
            self.compiled.save_weights(path + name + '.h5f', overwrite=True)
        else:
            self.compiled.model.save_weights(path + name + '.h5f', overwrite=True)

        if pickle_out:
            with open(path + name + '.pickle', 'wb') as f:
                pickle.dump([self.history, self.info], f)
                f.close()
