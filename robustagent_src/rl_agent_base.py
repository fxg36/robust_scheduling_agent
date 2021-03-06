from copy import deepcopy
from random import choice
import sys
import time
from typing import Callable, List
import gym
from gym.spaces import Box, Discrete
import numpy as np
import pandas as pd
import hyperparam as hp
from robustness_evaluator import BaselineSchedule, Result
import data as d
import flowshop_milp as milp
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import torch as th
from robustness_evaluator import simulate_deterministic_bs_schedule
import sample_management
from copy import deepcopy

NO_OBSERVATION_FEATURES = 14
NO_ACTIONS = 3  # = factor to stretch processing times. chosen by the agent for each task
NET_ARCH = dict(
    activation_fn=th.nn.ReLU, net_arch=[512, dict(vf=[128, 64], pi=[64])]
)  # just hidden layers. input and output layer are set automatically by stable baselines.


def get_env(n_steps, learning_rate_start=0, use_train_samples=False):
    samples = sample_management.load_samples(hp.SAMPLES_TO_LOAD, use_train_samples)
    env = RobustFlowshopGymEnv(samples, n_steps, learning_rate_start)
    env = Monitor(env, hp.TENSORBOARD_LOG_PATH, allow_early_resets=True)
    venv = DummyVecEnv([lambda: env])
    return VecNormalize(venv, training=True, norm_obs=True, norm_reward=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def get_model_name(algorithm_name: str, model_no=-1, with_model_path=False):
    name = f"{algorithm_name}_{hp.SCHED_OBJECTIVE}"
    name = f'{name}_n{model_no}' if model_no >= 0 else name
    res = {}
    res['model_name'] = name
    if with_model_path:
        p = Path(".")
        p = p / "drl_models" / f"model_{name}"
        res['model_path'] = p
    return res


def train(model, algorithm_name, trainsteps, model_no, save=True):
    model_info = get_model_name(algorithm_name, model_no, with_model_path=save)
    model.learn(total_timesteps=trainsteps, log_interval=1, tb_log_name=model_info['model_name'])
    print("#############################\ntraining completed")
    if save:
        model.save(model_info['model_path'])


def test(env_norm, model, model_name, result_file_suffix, test_episodes=100):
    Result.results = []
    durations = []
    obs = env_norm.reset()
    e = 0
    while True:
        start_time = time.time()
        action, _states = model.predict(np.array(obs))
        obs, reward, done, info = env_norm.step(action)
        if done:
            e += 1
            durations.append(time.time() - start_time)
            print(f"{e}/{test_episodes} TEST RESULT ({durations[-1]})")
            if e == test_episodes:
                break
            env_norm.reset()

    Result.write_results(model_name, result_file_suffix)


class RobustFlowshopGymEnv(gym.Env):
    def __init__(self, baseline_schedules: List[BaselineSchedule], n_steps, initial_learning_rate):
        self.action_space = Discrete(NO_ACTIONS)
        self.observation_space = Box(
            np.array([-sys.float_info.max] * NO_OBSERVATION_FEATURES),
            np.array([sys.float_info.max] * NO_OBSERVATION_FEATURES),
        )
        self.candidates = baseline_schedules
        self.curr_candidate = None
        self.curr_step_overall = 0
        self.n_steps_overall = n_steps
        self.initial_learning_rate = initial_learning_rate
        self._set_initial_state()

    def _set_initial_state(self):
        self.curr_candidate = choice(self.candidates)
        self.mc = self.curr_candidate.mc_stats
        self.modified_jobs = deepcopy(self.curr_candidate.job_dict)
        self.n_jobs = len(self.modified_jobs.values())
        self.n_tasks = self.n_jobs * 3
        self.curr_episode_step = 0
        self.action_log = []
        self.job_slack_log = []
        self.machine_slack_log = []
        jobs_p1 = len(list(filter(lambda x: x == "p1", map(lambda x: x[5], self.curr_candidate.job_dict.values()))))
        self.last_plan_robustness = self.mc["r_mean"]
        self.state_dict = {
            "n_jobs": self.n_jobs
        }
        self._update_state()

    def _update_state(self, last=False):
        fs = simulate_deterministic_bs_schedule(self.modified_jobs, self.curr_candidate.evaluator.initial_start_times)
        self.state_dict["plan_robustness"] = (fs.kpis[hp.SCHED_OBJECTIVE] - self.curr_candidate.evaluator.initial_objective_value) + self.mc["r_mean"]


        job_task = self.curr_candidate.task_order[self.curr_episode_step if not last else self.curr_episode_step-1]
        job = job_task[0]
        machine = job_task[1]
        

        self.state_dict["added_slack_job"] = (
            sum(map(lambda y: y["slack"], filter(lambda x: x["job"] == job, self.job_slack_log)))
        )
        self.state_dict["added_slack_machine_rel"] = (
            sum(map(lambda y: y["slack"], filter(lambda x: x["task"] == machine, self.machine_slack_log))) / self.n_jobs
        )

        successors = self.curr_candidate.task_order[self.curr_episode_step + 1 :]
        successors_same_machine = list(filter(lambda x: x[1] == machine, successors))
        self.state_dict["n_successors_machine"] = len(successors_same_machine) / self.n_jobs
        
        self.state_dict["job_end_diff"] = fs.get_stability_metrices()[0][job,3] - self.mc['com_mean_job'][job,3]
        self.state_dict["operation_end_diff"] = fs.get_stability_metrices()[0][job,machine] - self.mc['com_mean_job'][job,machine]

        machine_dummies = pd.get_dummies(pd.Series([1, 2, 3])).values
        machine_dummy = machine_dummies[machine - 1]
        for i in range(len(machine_dummy)):
            self.state_dict[f"machine_{i}"] = machine_dummy[i]
        self.state_dict["product"] = 0 if self.curr_candidate.job_dict[job][5] == "p1" else 1

        self.state_dict["operation_totalslack_above_avg"] = (
            1 if self.mc["total_slack_mean_job"][job_task] > self.mc["mean_total_slack"] else 0
        )
        self.state_dict["critical_path"] = 1 if self.mc["total_slack_mean_job"][job_task] == 0 else 0

        step_rel = (self.curr_episode_step) / self.n_tasks
        self.state_dict["step_progress"] = step_rel


        self.state = list(self.state_dict.values())
        assert len(self.state) == NO_OBSERVATION_FEATURES, str(
            f"this must be equal! {len(self.state)} != {NO_OBSERVATION_FEATURES}"
        )

    def perform_action(self, action, job_task):
        new_duration = 0

        values = d.JobFactory.preprocess_one_operation(self.curr_candidate.jobs_raw, job_task[0], job_task[1])
        default = values[2]
        very_optimistic = values[1]
        very_conservative = values[3]

        wr = hp.WEIGHT_ROBUSTNESS
        ws = 1-wr

        # use default value (expected duration incl. downtime)
        if action == 0:
            new_duration = default

        # use exptected value - stdev (optimistic)
        if action == 1:
            new_duration = very_optimistic
        
        # consider downtime and stdev (conservative)
        elif action == 2:
            new_duration = very_conservative

        return new_duration

    def step(self, action):
        job_task = self.curr_candidate.task_order[self.curr_episode_step]
        job = job_task[0]
        task = job_task[1]
        task_to_modify = next(filter(lambda x: isinstance(x, list) and x[0] == task, self.modified_jobs[job]))

        old_duration = task_to_modify[1]
        new_duration = self.perform_action(action, job_task)

        # stretch or extend task
        task_to_modify[1] = new_duration
        dur_diff = new_duration - old_duration
        self.job_slack_log.append({"job": job, "slack": -1 if dur_diff < 0 else 1 if dur_diff > 0 else 0})
        self.machine_slack_log.append({"task": task, "slack": -1 if dur_diff < 0 else 1 if dur_diff > 0 else 0})
        
        self.curr_episode_step += 1
        self.curr_step_overall += 1

        reward = 0
        if self.curr_episode_step == self.n_tasks:
            done = True
            reward += self.handle_episode_end(reward)
            self._update_state(last=True)
        else:
            done = False
            self._update_state()
        
        reward += self._get_inter_reward(action)
        
        info = {"reward": reward, "action_log": self.action_log}
        return self.state, reward, done, info

    def _get_inter_reward(self, action):
        wr = hp.WEIGHT_ROBUSTNESS
        ws = 1-wr
        ir = 0

        # use exptected value without downtime (optimistic)
        if action == 1:
            ir += 4*ws

        if(abs(self.state_dict["plan_robustness"])<abs(self.last_plan_robustness)):
            ir += 6*wr
        else:
            ir -= 4*ws

        self.last_plan_robustness = self.state_dict["plan_robustness"]
        self.action_log.append({'action':action, 'inter_reward':ir, 'obs': deepcopy(self.state_dict)})

        return ir

    def handle_episode_end(self, reward):
        v, stats, _ = self.curr_candidate.evaluator.eval(self.modified_jobs)

        reward += v ** 10 * -100 if v < 1 else v * -125

        actions = list(map(lambda x: x['action'], self.action_log))
        interim_rewards = sum(list(map(lambda x: x['inter_reward'], self.action_log)))
        action_std = np.std(actions)
        lr = self.initial_learning_rate * ((self.n_steps_overall - self.curr_step_overall) / self.n_steps_overall)
        obj = stats["makespan_mean"] if hp.SCHED_OBJECTIVE == milp.Objective.CMAX else stats["flowtime_mean"]
        print(
            f"Ep{self.curr_step_overall:4d}/{self.n_steps_overall} (LR={lr:10.8f})  |  Rob/Stab={v:4.3f}  |  EndRew={reward:6.2f}  |  InterRew={interim_rewards:5.3f}  |  ActionStd={action_std:4.3f} ({actions})  |  {obj}"
        )

        return reward

    def reset(self):
        self._set_initial_state()
        return self.state

    def render(self):
        pass  # not neccessary!