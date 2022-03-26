from copy import deepcopy
import pickle
from random import choice, randint
import sys
from typing import List
import gym
from gym.spaces import Box, Discrete
import numpy as np
import pandas as pd
import hyperparam as hp
from robustness_evaluator import RobustnessEvaluator
import data as d
import flowshop_milp as milp
import flowshhop_simulation as s
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

NO_OBSERVATION_FEATURES = 18  # describes the environment state each step
NO_ACTIONS = 3  # = factor to stretch processing times. chosen by the agent for each task
  # instances to be generated for training/testing
# no. of jobs for the test/training instances
EPISODE_LEN = hp.NO_JOBS * 3  # 3 machines = 3 tasks per job


class BaselineSchedule:
    def __init__(self, jobs_raw):
        d.JobFactory.preprocess_jobs(jobs_raw)
        self.jobs_raw = jobs_raw
        self.no_p1 = list(map(lambda x: int(x[5].replace("p", "")), jobs_raw)).count(1)
        start_times, obj_val = milp.solve(hp.SCHED_OBJECTIVE, jobs_raw)
        self.objective_value = obj_val
        self.job_dict = milp.get_job_dict(jobs_raw)
        self.task_order = list(map(lambda x: x[0], sorted(list(start_times.items()), key=lambda x: x[1])))
        self.mc_stats = s.run_monte_carlo_experiments(obj_val, self.job_dict, start_times, n_experiments=10000)
        self.evaluator = RobustnessEvaluator(jobs_raw)

        assert len(self.task_order) == EPISODE_LEN, "EPISODE LEN is not correct. Check no. machines/operations"
        assert len(jobs_raw) == hp.NO_JOBS, "NO JOBS is not set properly!"


def generate_samples(n_jobs):
    samples = []
    for _ in range(hp.SAMPLES):
        no_jobs_p1 = randint(1, hp.NO_JOBS - 1)
        random_jobvector = d.JobFactory.get_random_jobs(no_jobs_p1, hp.NO_JOBS - no_jobs_p1)
        bs = BaselineSchedule(random_jobvector)
        samples.append(bs)
        print("CREATED 1 SCHEDULE")
    with open(f"samples{hp.SAMPLES}_jobs{n_jobs}_{hp.SCHED_OBJECTIVE}.pickle", "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples(n_jobs):
    try:
        with open(f"samples{hp.SAMPLES}_jobs{n_jobs}_{hp.SCHED_OBJECTIVE}.pickle", "rb") as f:
            samples = pickle.load(f)
        return samples
    except:
        generate_samples(n_jobs)
        return load_samples(n_jobs)


def get_env(n_episodes, learning_rate_start):
    samples = load_samples(hp.NO_JOBS)
    env = RobustFlowshopGymEnv(samples, n_episodes, learning_rate_start)
    env = Monitor(env, hp.TENSORBOARD_LOG_PATH, allow_early_resets=True)
    venv = DummyVecEnv([lambda: env])
    return VecNormalize(venv, training=True, norm_obs=True, norm_reward=True)


class RobustFlowshopGymEnv(gym.Env):
    def __init__(self, baseline_schedules: List[BaselineSchedule], n_episodes, initial_learning_rate):
        self.action_space = Discrete(NO_ACTIONS)
        self.observation_space = Box(
            np.array([-sys.float_info.max] * NO_OBSERVATION_FEATURES),
            np.array([sys.float_info.max] * NO_OBSERVATION_FEATURES),
        )
        self.candidates = baseline_schedules
        self.curr_candidate = None
        self.curr_episode = 0
        self.n_episodes = n_episodes
        self.initial_learning_rate = initial_learning_rate
        self._set_initial_state()

    def _set_initial_state(self):
        self.curr_candidate = choice(self.candidates)
        self.modified_jobs = deepcopy(self.curr_candidate.job_dict)
        self.mc = self.curr_candidate.mc_stats
        self.episode_step = 0
        self.action_log = []
        self.slack_log = []
        self.state_dict = {
            "conservative_schedule": 1 if self.mc["r_mean"] > 0 else 0,
            "final_result": 0,
            "initial_makespan_flowtime_ratio": self.mc["makespan_mean"] / self.mc["flowtime_mean"],
            "initial_totalslack_makespan_ratio": sum(self.mc["total_slack_mean_job"].values())
            / self.mc["makespan_mean"],
            "initial_freeslack_makespan_ratio": sum(self.mc["free_slack_mean_job"].values())
            / self.mc["makespan_mean"],  # machine_wait_time_ratio_mean
        }
        self._update_state()

    def _update_state(self, last=False, stats=None, final_result=None):
        self.state_dict["additional_slack_rel"] = sum(self.slack_log) / self.mc["makespan_mean"]
        if not last:
            job_task = self.curr_candidate.task_order[self.episode_step]
            job = job_task[0]
            machine = job_task[1]
            machine_dummies = pd.get_dummies(pd.Series([1, 2, 3])).values
            machine_dummy = machine_dummies[machine - 1]
            for i in range(len(machine_dummy)):
                self.state_dict[f"machine_{i}"] = machine_dummy[i]
            self.state_dict["product"] = 0 if self.curr_candidate.job_dict[job][5] == "p1" else 1
            job_slack = self.mc["total_slack_mean_job"][job_task]
            job_slack_above_mean = job_slack > self.mc["mean_total_slack"]
            self.state_dict["no_totalslack"] = 1 if job_slack == 0 else 0
            self.state_dict["totalslack_above_mean"] = 1 if job_slack_above_mean else 0

            job_freeslack = self.mc["free_slack_mean_job"][job_task]
            job_freeslack_above_mean = job_freeslack > self.mc["mean_free_slack"]
            self.state_dict["no_freeslack"] = 1 if job_freeslack == 0 else 0
            self.state_dict["freeslack_above_mean"] = 1 if job_freeslack_above_mean else 0

            successors = self.curr_candidate.task_order[self.episode_step + 1 :]
            successors_same_machine = list(filter(lambda x: x[1] == machine, successors))
            self.state_dict["n_successors_machine"] = len(successors_same_machine) / hp.NO_JOBS

            step_rel = self.episode_step / EPISODE_LEN
            self.state_dict["wait_time_ratio_machine"] = self.mc["machine_wait_time_ratio_mean"][machine]
            self.state_dict["first_third_of_schedule"] = 1 if step_rel <= 1 / 3 else 0
            self.state_dict["second_third_of_schedule"] = 1 if step_rel > 1 / 3 and step_rel <= 2 / 3 else 0

        if last:
            self.state_dict["final_result"] = final_result

        self.state = list(self.state_dict.values())
        assert len(self.state) == NO_OBSERVATION_FEATURES, str(
            f"this must be equal! {len(self.state)} != {NO_OBSERVATION_FEATURES}"
        )

    def step(self, action):
        reward = 0
        job_task = self.curr_candidate.task_order[self.episode_step]
        job = job_task[0]
        task = job_task[1]
        task_to_modify = next(filter(lambda x: isinstance(x, list) and x[0] == task, self.modified_jobs[job]))
        task_dur_std = self.mc["dur_std_job"][job_task]
        assert task_dur_std >= 0, "std must be positive"

        low_slack = self.state_dict["totalslack_above_mean"] == 0
        high_slack = not low_slack
        no_slack = self.state_dict["no_totalslack"] == 1
        slack_to_append = 0

        if action == 0:
            if low_slack:  # but if the operation has a low slack: punish!
                if no_slack:
                    reward += -8 / EPISODE_LEN  # punish harder when it has no slack
                else:
                    reward += -4 / EPISODE_LEN

        if action == 1:  # extend operation time (=add slack)
            slack_to_append = task_dur_std / 8  # to add: quarter of operation time std
            if high_slack:  # but if the operation already has a high slack: punish!
                reward += -4 / EPISODE_LEN
        if action == 2:  # extend operation time (=add slack)
            slack_to_append = task_dur_std / 4  # to add: quarter of operation time std
            if high_slack:  # but if the operation already has a high slack: punish!
                reward += -8 / EPISODE_LEN

        task_to_modify[1] = int(round(task_to_modify[1] + slack_to_append))

        self.slack_log.append(slack_to_append)
        self.action_log.append([action, reward])

        self.episode_step += 1
        if self.episode_step == EPISODE_LEN:
            done = True
            reward += self.handle_episode_end()
        else:
            done = False
            self._update_state()

        info = {"reward": reward}
        return self.state, reward, done, info

    def handle_episode_end(self):
        v, stats = self.curr_candidate.evaluator.eval(self.modified_jobs)
        reward = -v * 64

        actions = list(map(lambda x: x[0], self.action_log))
        interim_rewards = sum(list(map(lambda x: x[1], self.action_log)))
        action_std = np.std(actions)

        lr = self.initial_learning_rate * ((self.n_episodes - self.curr_episode) / self.n_episodes)
        print(
            f"Ep{self.curr_episode:4d}/{self.n_episodes} (LR={lr:10.8f})  |  Rob/Stab={v:4.3f}  |  EndRew={reward:6.2f}  |  InterRew={interim_rewards:5.3f}  |  ActionStd={action_std:4.3f} ({actions})"
        )

        self._update_state(last=True, stats=stats, final_result=v)

        return reward

    def reset(self):
        self.curr_episode += 1
        self._set_initial_state()
        return self.state

    def render(self):
        pass  # not neccessary!


""" old env 
class RobustFlowshopGymEnv(gym.Env):
    def __init__(self, baseline_schedules: List[base.BaselineSchedule]):
        self.action_space = Discrete(base.NO_ACTIONS)
        self.observation_space = Box(
            np.array([-sys.float_info.max] * base.NO_OBSERVATION_FEATURES),
            np.array([sys.float_info.max] * base.NO_OBSERVATION_FEATURES),
        )
        self.candidates = baseline_schedules
        self.curr_candidate = None
        self.curr_episode = 0
        self._set_initial_state()

    def _set_initial_state(self):
        self.curr_candidate = choice(self.candidates)
        self.modified_jobs = deepcopy(self.curr_candidate.job_dict)
        self.mc = self.curr_candidate.mc_stats
        self.episode_step = 0
        self.action_log = []
        self.slack_log = []
        self.state_dict = {
            #"initial_objective": self.curr_candidate.objective_value,
            #"n_jobs": len(self.curr_candidate.jobs_raw),
            #"n_jobs_p1": self.curr_candidate.no_p1,
            #"start_robustness": self.mc["r_mean"],
            #"end_robustness": self.mc["r_mean"],
            'conservative_schedule': 1 if self.mc["r_mean"] > 0 else 0,
            'final_result': 0,
            #"start_stability": self.mc["scom_mean"],
            #"end_stability": self.mc["scom_mean"],
            #"initial_mean_totalslack": self.mc["mean_total_slack"],
            #"initial_mean_freeslack": self.mc["mean_free_slack"],
            # "initial_totalslack": sum(self.mc["total_slack_mean_job"].values()),
            # "initial_freeslack": sum(self.mc["free_slack_mean_job"].values()),
            # #"end_mean_slack": self.mc["mean_total_slack"],
            # "initial_makespan": self.mc['makespan_mean'],
            # "initial_flowtime": self.mc['flowtime_mean'],
            "initial_makespan_flowtime_ratio": self.mc['makespan_mean'] / self.mc['flowtime_mean'],
            "initial_totalslack_makespan_ratio": sum(self.mc["total_slack_mean_job"].values()) / self.mc['makespan_mean'],
            "initial_freeslack_makespan_ratio": sum(self.mc["free_slack_mean_job"].values()) / self.mc['makespan_mean'], #machine_wait_time_ratio_mean
        }
        self._update_state()

    def _update_state(self, last=False, stats=None, final_result=None):
        self.state_dict["additional_slack_rel"] = sum(self.slack_log) / self.mc['makespan_mean']
        if not last:
            job_task = self.curr_candidate.task_order[self.episode_step]
            job = job_task[0]
            machine = job_task[1]
            machine_dummies = pd.get_dummies(pd.Series([1, 2, 3])).values
            machine_dummy = machine_dummies[machine - 1]
            for i in range(len(machine_dummy)):
                self.state_dict[f"machine_{i}"] = machine_dummy[i]
            self.state_dict["product"] = 0 if self.curr_candidate.job_dict[job][5] == "p1" else 1
            # self.state_dict['due_date'] = self.curr_candidate.job_dict[job][3]
            job_slack = self.mc["total_slack_mean_job"][job_task]
            job_slack_above_mean = job_slack > self.mc["mean_total_slack"]
            self.state_dict["no_totalslack"] = 1 if job_slack == 0 else 0
            self.state_dict["totalslack_above_mean"] = 1 if job_slack_above_mean else 0
            
            job_freeslack = self.mc["free_slack_mean_job"][job_task]
            job_freeslack_above_mean = job_freeslack > self.mc["mean_free_slack"]
            self.state_dict["no_freeslack"] = 1 if job_freeslack == 0 else 0
            self.state_dict["freeslack_above_mean"] = 1 if job_freeslack_above_mean else 0

            successors = self.curr_candidate.task_order[self.episode_step+1:]
            #self.state_dict["n_successors"] = len(successors) / EPISODE_LEN
            successors_same_machine = list(filter(lambda x: x[1] == machine, successors))
            self.state_dict["n_successors_machine"] = len(successors_same_machine) / hp.NO_JOBS

            step_rel = self.episode_step / base.EPISODE_LEN
            self.state_dict["wait_time_ratio_machine"] = self.mc['machine_wait_time_ratio_mean'][machine]
            self.state_dict['first_third_of_schedule'] = 1 if step_rel <= 1/3 else 0
            self.state_dict['second_third_of_schedule'] = 1 if step_rel > 1/3 and step_rel <= 2/3 else 0

            #self.state_dict["total_slack_mean_job"] = self.mc["total_slack_mean_job"][job_task]
            #self.state_dict["free_slack_mean_job"] = self.mc["free_slack_mean_job"][job_task]
            
            
            
            
            # self.state_dict["total_slack_std_job"] = self.mc["total_slack_std_job"][job_task]
            # self.state_dict['dur_mean_job'] = self.mc["dur_mean_job"][job_task]
            # self.state_dict['com_mean_job'] = self.mc["com_mean_job"][job_task]
            # self.state_dict['dur_std_job'] = self.mc["dur_std_job"][job_task]
            # self.state_dict['com_std_job'] = self.mc["com_std_job"][job_task]
            # planned_duration = self.curr_candidate.job_dict[job][machine - 1][1]
            # self.state_dict['planned_duration'] = planned_duration
        if last:
            # self.state_dict["end_robustness"] = stats["r_mean"]
            # self.state_dict["end_stability"] = stats["scom_mean"]
            # self.state_dict["end_mean_slack"] = stats["mean_total_slack"]
            self.state_dict["final_result"] = final_result

        self.state = list(self.state_dict.values())
        assert len(self.state) == base.NO_OBSERVATION_FEATURES, str(
            f"this must be equal! {len(self.state)} != {base.NO_OBSERVATION_FEATURES}"
        )

    def step(self, action):
        reward = 0
        job_task = self.curr_candidate.task_order[self.episode_step]
        job = job_task[0]
        task = job_task[1]
        task_to_modify = next(filter(lambda x: isinstance(x, list) and x[0] == task, self.modified_jobs[job]))
        task_dur_std = self.mc["dur_std_job"][job_task]
        assert task_dur_std >= 0, "std must be positive"

        conservative = self.state_dict['conservative_schedule'] == 1
        low_slack = self.state_dict["totalslack_above_mean"] == 0
        high_slack = not low_slack
        no_slack = self.state_dict["no_totalslack"] == 1
        slack_to_append = 0

        # if action == 0:  # do not stretch or compress the operation time
        #     if no_slack and hp.WEIGHT_ROBUSTNESS > 0.5:
        #         # if the operation has no slack and it is priority to optimize robustness
        #         # -> punish
        #         reward += -1 / EPISODE_LEN
        #     elif high_slack and (1 - hp.WEIGHT_ROBUSTNESS) > 0.5:  # ????
        #         # if the operation has high slack and it is priority to optimize stability
        #         # -> punish
        #         reward += -1 / EPISODE_LEN

        if action == 0:
            if low_slack:  # but if the operation has a low slack: punish!
                if no_slack:
                    reward += -8 / base.EPISODE_LEN  # punish harder when it has no slack
                else:
                    reward += -4 / base.EPISODE_LEN

        if action == 1:  # extend operation time (=add slack)
            # The less important stability is, the more important it is to extend op. time.
            # -> Punish lightly when times are longer while optimizing robustness
            #reward = (-2 / EPISODE_LEN) * (1 - hp.WEIGHT_ROBUSTNESS)

            slack_to_append = task_dur_std / 8  # to add: quarter of operation time std

            if high_slack:  # but if the operation already has a high slack: punish!
                reward += -4 / base.EPISODE_LEN

        if action == 2:  # extend operation time (=add slack)
            # The less important stability is, the more important it is to extend op. time.
            # -> Punish lightly when times are longer while optimizing robustness
            #reward = (-2 / EPISODE_LEN) * (1 - hp.WEIGHT_ROBUSTNESS)

            slack_to_append = task_dur_std / 4  # to add: quarter of operation time std

            if high_slack:  # but if the operation already has a high slack: punish!
                reward += -8 / base.EPISODE_LEN

        # if action == 2:  # compress operation time (=remove slack)
        #     # The less important robustness is, the more important it is to shorten op. time.
        #     # -> Punish lightly when times are shortened to optimize stability
        #     #reward = (-1 / EPISODE_LEN) * hp.WEIGHT_ROBUSTNESS

        #     slack_to_append = -task_dur_std / 4  # to remove: quarter of operation time std

        #     if low_slack:  # but if the operation has a low slack: punish!
        #         if no_slack:
        #             reward += -4 / EPISODE_LEN  # punish harder when it has no slack
        #         else:
        #             reward += -2 / EPISODE_LEN
        #     if not conservative:
        #         reward += -2 / EPISODE_LEN

        task_to_modify[1] = int(round(task_to_modify[1] + slack_to_append))
        # task_to_modify[1] = int(round(self.mc["dur_mean_job"][job_task] + slack_to_append))

        self.slack_log.append(slack_to_append)
        self.action_log.append([action, reward])

        self.episode_step += 1
        if self.episode_step == base.EPISODE_LEN:
            done = True
            reward += self.handle_episode_end()
        else:
            done = False
            self._update_state()

        info = {"reward": reward}
        return self.state, reward, done, info

    def handle_episode_end(self):
        v, stats = self.curr_candidate.evaluator.eval(self.modified_jobs)
        # reward = (-3**v + 1) * 50
        reward = -v * 64

        actions = list(map(lambda x: x[0], self.action_log))
        interim_rewards = sum(list(map(lambda x: x[1], self.action_log)))
        action_std = np.std(actions)

        lr = LEARNING_RATE_START * ((EPISODES - self.curr_episode) / EPISODES)
        print(
            f"Ep{self.curr_episode:4d}/{EPISODES} (LR={lr:10.8f})  |  Rob/Stab={v:4.3f}  |  EndRew={reward:6.2f}  |  InterRew={interim_rewards:5.3f}  |  ActionStd={action_std:4.3f} ({actions})"
        )

        self._update_state(last=True, stats=stats, final_result=v)

        return reward

    def reset(self):
        self.curr_episode += 1
        self._set_initial_state()
        return self.state

    def render(self):
        pass  # not neccessary!
"""