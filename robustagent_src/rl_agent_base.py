from copy import deepcopy
import pickle
from random import choice, randint
import sys
import time
from typing import Callable, List
import gym
from gym.spaces import Box, Discrete
import numpy as np
import pandas as pd
import hyperparam as hp
from robustness_evaluator import BaselineSchedule
import data as d
import flowshop_milp as milp
import flowshhop_simulation as s
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

NO_OBSERVATION_FEATURES = 18
NO_ACTIONS = 3  # = factor to stretch processing times. chosen by the agent for each task

V = []


def generate_samples(n_jobs):
    samples = []
    for _ in range(hp.N_SAMPLES):
        no_jobs_p1 = randint(1, n_jobs - 1)
        random_jobvector = d.JobFactory.get_random_jobs(no_jobs_p1, n_jobs - no_jobs_p1)
        bs = BaselineSchedule(random_jobvector)
        samples.append(bs)
        print("CREATED 1 SCHEDULE")

    p = Path(".")
    p = p / "samples" / f"samples{hp.N_SAMPLES}_jobs{n_jobs}_{hp.SCHED_OBJECTIVE}.pickle"
    with open(p, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples(n=hp.SAMPLES_TO_LOAD):
    if n == 0:
        samples = []
        samples.extend(load_samples(4))
        samples.extend(load_samples(6))
        samples.extend(load_samples(8))
        return samples
    else:
        p = Path(".")
        p = p / "samples" / f"samples{hp.N_SAMPLES}_jobs{n}_{hp.SCHED_OBJECTIVE}.pickle"
        try:
            with open(p, "rb") as f:
                samples = pickle.load(f)
            return samples
        except:
            generate_samples(n)
            return load_samples(n)


def get_env(n_steps, learning_rate_start=0):
    samples = load_samples()
    env = RobustFlowshopGymEnv(samples, n_steps, learning_rate_start)
    env = Monitor(env, hp.TENSORBOARD_LOG_PATH, allow_early_resets=True)
    venv = DummyVecEnv([lambda: env])
    return VecNormalize(venv, training=True, norm_obs=True, norm_reward=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def train(model, name, trainsteps, save=True):
    name = f"{name}_{hp.SCHED_OBJECTIVE}_J{hp.N_JOBS}"
    model.learn(total_timesteps=trainsteps, log_interval=1, tb_log_name=name)
    print("#############################\ntraining completed")
    if save:
        p = Path(".")
        p = p / "drl_models" / f"model_{name}"
        model.save(p)


def test(env_norm, model, test_episodes=100):
    V = []
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
            print(f"TEST RESULT ({durations[-1]})")
            if e == test_episodes:
                break
            env_norm.reset()
    for v in V:
        print(v)
    print(f"min: {min(V)}")
    print(f"mean: {np.mean(V)}")
    print(f"std: {np.std(V)}")
    return V, min(V), np.mean(V), np.std(V), durations, np.mean(durations)


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
        # if len(V) < 10 or V[-1] < 1 or self.curr_step_overall/self.n_steps_overall > 0.5:
        #    self.curr_candidate = choice(self.candidates)
        self.curr_candidate = choice(self.candidates)
        self.mc = self.curr_candidate.mc_stats
        self.modified_jobs = deepcopy(self.curr_candidate.job_dict)
        self.n_jobs = len(self.modified_jobs.values())
        self.n_tasks = self.n_jobs * 3
        self.curr_episode_step = 0
        self.action_log = []
        self.slack_log = []
        jobs_p1 = len(list(filter(lambda x: x == "p1", map(lambda x: x[5], self.curr_candidate.job_dict.values()))))
        self.state_dict = {
            "conservative_schedule": 1 if self.mc["r_mean"] > 0 else 0,
            # "initial_makespan_flowtime_ratio": self.mc["makespan_mean"] / self.mc["flowtime_mean"],
            # "initial_totalslack_makespan_ratio": sum(self.mc["total_slack_mean_job"].values())
            # / self.mc["makespan_mean"],
            # "initial_freeslack_makespan_ratio": sum(self.mc["free_slack_mean_job"].values())
            # / self.mc["makespan_mean"],  # machine_wait_time_ratio_mean
            "n_jobs": self.n_jobs,
            "products_std": np.std([jobs_p1, self.n_jobs - jobs_p1]),
            # "r_start": abs(self.mc["r_mean"]) * self.n_tasks,
            # "s_start": self.mc["scom_mean"] / self.n_tasks,
            "flowtime_rel": self.mc["flowtime_mean"] / self.n_tasks,
            "makespan_rel": self.mc["makespan_mean"] / self.n_tasks,
            "totalslack_mean": self.mc["mean_total_slack"],
            "freeslack_mean": self.mc["mean_free_slack"],
            "final_result": 0,
        }
        self._update_state()

    def _update_state(self, last=False, stats=None, final_result=None):
        self.state_dict["additional_slack_rel"] = sum(self.slack_log) / self.n_tasks
        if not last:
            job_task = self.curr_candidate.task_order[self.curr_episode_step]
            job = job_task[0]
            machine = job_task[1]
            machine_dummies = pd.get_dummies(pd.Series([1, 2, 3])).values
            machine_dummy = machine_dummies[machine - 1]
            for i in range(len(machine_dummy)):
                self.state_dict[f"machine_{i}"] = machine_dummy[i]
            self.state_dict["product"] = 0 if self.curr_candidate.job_dict[job][5] == "p1" else 1

            job_slack = self.mc["total_slack_mean_job"][job_task]
            # job_slack_above_mean = job_slack > self.mc["mean_total_slack"]
            self.state_dict["totalslack"] = job_slack
            # self.state_dict["totalslack_above_mean"] = 1 if job_slack_above_mean else 0
            # self.state_dict["critical_path"] = 1 if job_slack == 0 else 0

            job_freeslack = self.mc["free_slack_mean_job"][job_task]
            # job_freeslack_above_mean = job_freeslack > self.mc["mean_free_slack"]
            self.state_dict["freeslack"] = job_freeslack
            # self.state_dict["no_freeslack"] = 1 if job_freeslack == 0 else 0
            # self.state_dict["freeslack_above_mean"] = 1 if job_freeslack_above_mean else 0

            successors = self.curr_candidate.task_order[self.curr_episode_step + 1 :]
            successors_same_machine = list(filter(lambda x: x[1] == machine, successors))
            self.state_dict["n_successors_machine"] = len(successors_same_machine) / self.n_jobs

            step_rel = (self.curr_episode_step + 1) / self.n_tasks
            self.state_dict["step_progress"] = step_rel

            self.state_dict["wait_time_ratio_machine"] = self.mc["machine_wait_time_ratio_mean"][machine]
            # self.state_dict["first_quarter_of_schedule"] = 1 if step_rel <= 1 / 4 else 0
            # self.state_dict["second_quarter_of_schedule"] = 1 if step_rel > 1 / 4 and step_rel <= 2 / 4 else 0
            # self.state_dict["third_quarter_of_schedule"] = 1 if step_rel > 2 / 4 and step_rel <= 3 / 4 else 0
            # self.state_dict["last_operation_slack_added"] = 0 if len(self.slack_log) < 2 or self.slack_log[-2] == 0 else 1

        if last:
            self.state_dict["final_result"] = final_result
            # self.state_dict["r_final"] = abs(stats['r_mean']) / self.mc["makespan_mean"]
            # self.state_dict["s_final"] = stats['scom_mean'] / self.mc["makespan_mean"]

        self.state = list(self.state_dict.values())
        assert len(self.state) == NO_OBSERVATION_FEATURES, str(
            f"this must be equal! {len(self.state)} != {NO_OBSERVATION_FEATURES}"
        )

    def step(self, action):
        reward = 0
        job_task = self.curr_candidate.task_order[self.curr_episode_step]
        job = job_task[0]
        task = job_task[1]
        task_to_modify = next(filter(lambda x: isinstance(x, list) and x[0] == task, self.modified_jobs[job]))
        task_dur_std = self.mc["dur_std_job"][job_task]
        assert task_dur_std >= 0, "std must be positive"

        low_slack = self.mc["total_slack_mean_job"][job_task] < self.mc["mean_total_slack"]
        critical_path = self.mc["total_slack_mean_job"][job_task] == 0
        high_slack = not low_slack
        slack_to_append = 0
        if action == 0:
            if low_slack:  # but if the operation has a low slack: punish!
                if critical_path:
                    reward += -5 / self.n_tasks  # punish harder when it has no slack
                else:
                    reward += -2.5 / self.n_tasks
        if action == 1:  # extend operation time (=add slack)
            slack_to_append = task_dur_std / 5  # to add: quarter of operation time std
            if high_slack:  # but if the operation already has a high slack: punish!
                reward += -7.5 / self.n_tasks
        if action == 2:  # compress operation time (=remove slack)
            slack_to_append = -task_dur_std / 5
            if low_slack:
                if critical_path:
                    reward += -10 / self.n_tasks
                else:
                    reward += -5 / self.n_tasks
        # if action == 2:  # extend operation time (=add little slack)
        #     slack_to_append = task_dur_std / 10  # to add: quarter of operation time std
        #     if high_slack:  # but if the operation already has a high slack: punish!
        #         reward += -4 / self.n_tasks

        task_to_modify[1] = int(round(task_to_modify[1] + slack_to_append))

        self.slack_log.append(slack_to_append)
        self.action_log.append([action, reward])

        self.curr_episode_step += 1
        self.curr_step_overall += 1

        if self.curr_episode_step == self.n_tasks:
            done = True
            reward += self.handle_episode_end()
            info = {"reward": reward, "RSV": V[-1]}
        else:
            done = False
            info = {"reward": reward}
            self._update_state()

        return self.state, reward, done, info

    def handle_episode_end(self):
        v, stats = self.curr_candidate.evaluator.eval(self.modified_jobs)
        obj = stats["makespan_mean"] if hp.SCHED_OBJECTIVE == milp.Objective.CMAX else stats["flowtime_mean"]
        V.append(v)

        y = v ** 10 if v < 1 else v * 1.25
        reward = y * -100

        actions = list(map(lambda x: x[0], self.action_log))
        interim_rewards = sum(list(map(lambda x: x[1], self.action_log)))
        action_std = np.std(actions)

        lr = self.initial_learning_rate * ((self.n_steps_overall - self.curr_step_overall) / self.n_steps_overall)
        print(
            f"Ep{self.curr_step_overall:4d}/{self.n_steps_overall} (LR={lr:10.8f})  |  Rob/Stab={v:4.3f}  |  EndRew={reward:6.2f}  |  InterRew={interim_rewards:5.3f}  |  ActionStd={action_std:4.3f} ({actions})  |  {obj}"
        )

        self._update_state(last=True, stats=stats, final_result=v)

        return reward

    def reset(self):
        self._set_initial_state()
        return self.state

    def render(self):
        pass  # not neccessary!