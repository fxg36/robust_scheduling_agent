from typing import Callable
import numpy as np
from stable_baselines3 import A2C
import torch as th
import hyperparam as hp
import rl_agent_base as base

LEARNING_RATE_START = 0.0004  # controls how much to change the model in response to the estimated error each time the model weights are updated
DISCOUNT_FACTOR = 0.985  # how much the reinforcement learning agents cares about rewards in the distant future
EPISODES = 500  # no. of episodes for the training
NET_ARCH = dict(
    activation_fn=th.nn.ReLU, net_arch=[1024, dict(vf=[512, 128, 64, 32], pi=[256, 64, 16])]
)  # just hidden layers. input and output layer are set automatically by stable baselines.
STEPS_TO_UPDATE = 1 * base.EPISODE_LEN  # after this no. of steps, the weights are updated


def train():
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    env_norm = base.get_env(n_episodes=EPISODES, learning_rate_start=LEARNING_RATE_START)
    model = A2C(
        "MlpPolicy",  # actor critic
        env_norm,
        tensorboard_log=hp.TENSORBOARD_LOG_PATH,
        policy_kwargs=NET_ARCH,
        verbose=1,
        device="cuda",
        learning_rate=linear_schedule(LEARNING_RATE_START),
        gamma=DISCOUNT_FACTOR,
        n_steps=STEPS_TO_UPDATE,
        #batch_size=STEPS_TO_UPDATE,
    )
    model.learn(total_timesteps=base.EPISODE_LEN * EPISODES, log_interval=1, tb_log_name="ppo")
    model.save(f"model_a2c_{hp.SCHED_OBJECTIVE}_J{hp.NO_JOBS}")
    print("#############################\ntraining completed")


def test():
    env_norm = base.get_env(n_episodes=EPISODES, learning_rate_start=LEARNING_RATE_START)
    model = A2C.load(f"model_a2c_{hp.SCHED_OBJECTIVE}_J{4}")
    obs = env_norm.reset()
    eps = 100
    e = 0
    while True:
        action, _states = model.predict(np.array(obs))
        obs, reward, done, info = env_norm.step(action)
        if done:
            e += 1
            if e == eps:
                break
            print("TEST RESULT")
            env_norm.reset()
    for v in base.V:
        print(v)

perform_tests = 1
if perform_tests == 1:
    test()
else:
    train()
