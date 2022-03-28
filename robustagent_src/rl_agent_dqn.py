from typing import Callable
import numpy as np
from stable_baselines3 import DQN
import torch as th
import hyperparam as hp
import rl_agent_base as base

LEARNING_RATE_START = 0.0001  # controls how much to change the model in response to the estimated error each time the model weights are updated
DISCOUNT_FACTOR = 0.95  # how much the reinforcement learning agents cares about rewards in the distant future
EPISODES = 500  # no. of episodes for the training
# NET_ARCH = dict(activation_fn=th.nn.ReLU, net_arch=[625, dict(vf=[400,200,100,50,20,10], pi=[50,10])]) # just hidden layers. input and output layer are set automatically by stable baselines.
# NET_ARCH = dict(
#     activation_fn=th.nn.ReLU, net_arch=[128, dict(vf=[128, 64, 32, 16, 8], pi=[32, 16, 8])]
# )  # just hidden layers. input and output layer are set automatically by stable baselines.
NET_ARCH = dict(
    activation_fn=th.nn.ReLU, net_arch=[1024, 512, 256, 128, 64,32,16,8]
)  # just hidden layers. input and output layer are set automatically by stable baselines.
STEPS_TO_UPDATE = 1 * base.EPISODE_LEN  # after this no. of steps, the weights are updated


def train():
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    env_norm = base.get_env(n_episodes=EPISODES, learning_rate_start=LEARNING_RATE_START)
    model = DQN(
        "MlpPolicy",
        env_norm,
        tensorboard_log=hp.TENSORBOARD_LOG_PATH,
        policy_kwargs=NET_ARCH,
        verbose=1,
        device="cuda",
        learning_rate=linear_schedule(LEARNING_RATE_START),
        gamma=DISCOUNT_FACTOR,
        #n_steps=STEPS_TO_UPDATE,
        batch_size=STEPS_TO_UPDATE,
        learning_starts=0,
        exploration_fraction=0.1
    )
    model.learn(total_timesteps=base.EPISODE_LEN * EPISODES, log_interval=1, tb_log_name="ppo")
    model.save("model_ppo")
    print("#############################\ntraining completed")


def test():
    env_norm = base.get_env(n_episodes=EPISODES, learning_rate_start=LEARNING_RATE_START)
    model = DQN.load("model_ppo")
    obs = env_norm.reset()
    while True:
        action, _states = model.predict(np.array(obs))
        obs, reward, done, info = env_norm.step(action)
        if done:
            print("TEST RESULT")
            env_norm.reset()

perform_tests = 0
if perform_tests == 1:
    test()
else:
    train()
