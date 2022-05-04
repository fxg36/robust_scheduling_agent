from stable_baselines3 import A2C
import hyperparam as hp
import rl_agent_base as base

def train(lr_start: float, gamma: float, training_steps: int, steps_per_update: int, model_no=-1):
    env_norm = base.get_env(training_steps, lr_start, use_train_samples=True)
    model = A2C(
        "MlpPolicy",  # actor critic
        env_norm,
        tensorboard_log=hp.TENSORBOARD_LOG_PATH,
        policy_kwargs=base.NET_ARCH,
        verbose=1,
        device="cuda",
        learning_rate=base.linear_schedule(lr_start),
        gamma=gamma,
        n_steps=steps_per_update, 
        use_rms_prop=False
    )
    base.train(model, "a2c", training_steps, model_no)


def test(test_episodes: int, result_suffix: str, model_no=-1):
    env_norm = base.get_env(n_steps=test_episodes)
    model_info = base.get_model_name("a2c", model_no, with_model_path=True)
    base.test(env_norm, A2C.load(model_info['model_path']), model_info['model_name'], result_suffix, test_episodes)