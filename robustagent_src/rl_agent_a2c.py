from stable_baselines3 import A2C
import torch as th
import hyperparam as hp
import rl_agent_base as base

LEARNING_RATE_START = 0.005  # controls how much to change the model in response to the estimated error each time the model weights are updated
DISCOUNT_FACTOR = 0.97  # how much the reinforcement learning agents cares about rewards in the distant future


# NET_ARCH = dict(
#     activation_fn=th.nn.ReLU, net_arch=[1024, dict(vf=[512, 128, 64, 32], pi=[256, 64, 16])]
# )  # just hidden layers. input and output layer are set automatically by stable baselines.
NET_ARCH = dict(
    activation_fn=th.nn.ReLU, net_arch=[1024, dict(vf=[128], pi=[64])]
)  # just hidden layers. input and output layer are set automatically by stable baselines.

def train(lr_start: float, gamma: float, training_steps: int, steps_per_update: int):
    env_norm = base.get_env(training_steps, lr_start)
    model = A2C(
        "MlpPolicy",  # actor critic
        env_norm,
        tensorboard_log=hp.TENSORBOARD_LOG_PATH,
        policy_kwargs=NET_ARCH,
        verbose=1,
        device="cuda",
        learning_rate=base.linear_schedule(LEARNING_RATE_START),
        gamma=gamma,
        n_steps=steps_per_update, 
        use_rms_prop=False
    )
    base.train(model, 'a2c', training_steps)


def test(test_steps: int):
    env_norm = base.get_env(n_steps=test_steps)
    if hp.N_JOBS == 0:
        model = A2C.load(f"model_a2c_{hp.SCHED_OBJECTIVE}_J{0}")
    else:
        model = A2C.load(f"model_a2c_{hp.SCHED_OBJECTIVE}_J{4}")
    base.test(env_norm, model, test_steps)

# perform_tests = 1
# if perform_tests == 1:
#     test()
# else:
#     train()
