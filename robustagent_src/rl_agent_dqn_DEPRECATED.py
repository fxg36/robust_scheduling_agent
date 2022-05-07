# from typing import Callable
# import numpy as np
# from stable_baselines3 import DQN
# import torch as th
# import hyperparam as hp
# import rl_agent_base as base

# LEARNING_RATE_START = 0.0001  # controls how much to change the model in response to the estimated error each time the model weights are updated
# DISCOUNT_FACTOR = 0.95  # how much the reinforcement learning agents cares about rewards in the distant future
# TRAJECTORY_COLLECTING_EPISODES = 250
# TRAIN_EPISODES = 500  # no. of episodes for the training
# # NET_ARCH = dict(activation_fn=th.nn.ReLU, net_arch=[625, dict(vf=[400,200,100,50,20,10], pi=[50,10])]) # just hidden layers. input and output layer are set automatically by stable baselines.
# # NET_ARCH = dict(
# #     activation_fn=th.nn.ReLU, net_arch=[128, dict(vf=[128, 64, 32, 16, 8], pi=[32, 16, 8])]
# # )  # just hidden layers. input and output layer are set automatically by stable baselines.
# NET_ARCH = dict(
#     activation_fn=th.nn.ReLU, net_arch=[256,256,64,8]
# )  # just hidden layers. input and output layer are set automatically by stable baselines.
# STEPS_TO_UPDATE = hp.N_JOBS * 3 if hp.N_JOBS > 0 else 24   # after this no. of steps, the weights are updated
# EXPLORATION_PERIOD = 0.3
# EXPLORATION_VALUE_START = 1
# EXPLORATION_VALUE_END = 0.05

# def train():
#     def linear_schedule(initial_value: float) -> Callable[[float], float]:
#         def func(progress_remaining: float) -> float:
#             return progress_remaining * initial_value
#         return func

#     env_norm = base.get_env(n_steps=TRAIN_EPISODES+TRAJECTORY_COLLECTING_EPISODES, learning_rate_start=LEARNING_RATE_START)
#     model = DQN(
#         "MlpPolicy",
#         env_norm,
#         tensorboard_log=hp.TENSORBOARD_LOG_PATH,
#         policy_kwargs=NET_ARCH,
#         verbose=1,
#         device="cuda",
#         learning_rate=linear_schedule(LEARNING_RATE_START),
#         gamma=DISCOUNT_FACTOR,
#         #n_steps=STEPS_TO_UPDATE,
#         batch_size=STEPS_TO_UPDATE,
#         learning_starts=TRAJECTORY_COLLECTING_EPISODES * 3 * hp.N_JOBS,
#         exploration_fraction=EXPLORATION_PERIOD,
#         exploration_final_eps=EXPLORATION_VALUE_END,
#         exploration_initial_eps=EXPLORATION_VALUE_START
#     )
#     model.learn(total_timesteps=(hp.N_JOBS * 3 if hp.N_JOBS > 0 else 24) * TRAIN_EPISODES, log_interval=1, tb_log_name="dqn")
#     model.save(f"model_dqn_{hp.SCHED_OBJECTIVE}_J{hp.N_JOBS}")

#     print("#############################\ntraining completed")


# def test():
#     env_norm = base.get_env(n_steps=TRAIN_EPISODES, learning_rate_start=LEARNING_RATE_START)
#     if hp.N_JOBS == 0:
#         model = DQN.load(f"model_dqn_{hp.SCHED_OBJECTIVE}_J{0}_2")
#     else:
#         model = DQN.load(f"model_dqn_{hp.SCHED_OBJECTIVE}_J{4}")
#     obs = env_norm.reset()
#     eps = 100
#     e = 0
#     while True:
#         action, _states = model.predict(np.array(obs))
#         obs, reward, done, info = env_norm.step(action)
#         if done:
#             e += 1
#             if e == eps:
#                 break
#             print("TEST RESULT")
#             env_norm.reset()
#     for v in base.V:
#         print(v)

# perform_tests = 0
# if perform_tests == 1:
#     test()
# else:
#     train()
