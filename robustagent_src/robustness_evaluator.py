from pathlib import Path
from re import S
import flowshop_milp as milp
import flowshhop_simulation as sim
import hyperparam as hp
import data as d
import pandas as pd


def simulate_deterministic_bs_schedule(job_dict, start_times):
    fs = sim.FlowshopSimulation(
        job_dict,
        start_times,
        fire_dynamic_events=False,
    )
    fs.env.run(until=fs.meta_proc)
    return fs

def get_initial_stochastic_infos(obj_value, job_dict, start_times):
    return sim.run_monte_carlo_experiments(
            obj_value, hp.SCHED_OBJECTIVE, job_dict, start_times, n_experiments=10000
        )

class Result:
    results = []
    def __init__(self, rsv, s, r, s_base, r_base, time=None):
        self.rsv = rsv
        self.s = s
        self.r = r
        self.s_base = s_base
        self.r_base = r_base
        self.time = time
    

    @staticmethod
    def write_results(filename, suffix, reset=True):
        data = {
            'rsv':list(map(lambda x: x.rsv, Result.results)),
            'r':list(map(lambda x: x.r, Result.results)),
            'r_base':list(map(lambda x: x.r_base, Result.results)),
            'r_diff': list(map(lambda x: abs(x.r_base)-abs(x.r), Result.results)),
            's':list(map(lambda x: x.s, Result.results)),
            's_base':list(map(lambda x: x.s_base, Result.results)),
            's_diff': list(map(lambda x: x.s_base-x.s, Result.results)),
        }
        cols = ['rsv', 'r', 'r_base', 'r_diff', 's', 's_base', 's_diff']
        if Result.results[0].time != None:
            data['time'] = list(map(lambda x: x.time, Result.results))
            cols.append('time')

        df = pd.DataFrame(data)
        df = df[cols] 
        
        main_path = Path(".") / "experiment_results" / f"{filename}_{suffix}.csv"
        df.to_csv(main_path, sep=';', encoding='utf-8', header=True, index=False, decimal=",")
        
        descr = df.describe()
        stats_path = Path(".") / "experiment_results" / f"{filename}_{suffix}_stats.csv"
        descr.to_csv(stats_path, sep=';', encoding='utf-8', header=True, index=True, decimal=",")
        
        print(df)
        print(descr)

        if reset:
            Result.results = []

class RobustnessEvaluator:
    def __init__(self, jobs_raw, start_times, mc_stats_initial, objective_value_initial):
        self.job_dict = milp.get_job_dict(jobs_raw)
        assert (
            hp.WEIGHT_ROBUSTNESS >= 0 and hp.WEIGHT_ROBUSTNESS <= 1
        ), "WEIGHT_ROBUSTNESS is greater 1 or lower 0. This is not allowed!"
        self.initial_start_times = start_times
        self.initial_robustness = mc_stats_initial["r_mean"]
        self.initial_stability = mc_stats_initial["scom_mean"]
        self.initial_stats = mc_stats_initial
        self.initial_objective_value = objective_value_initial

    @staticmethod
    def calc_fittness(r, s, r_base, s_base, log_result=True):
        wr = hp.WEIGHT_ROBUSTNESS
        assert (
            hp.WEIGHT_ROBUSTNESS >= 0 and hp.WEIGHT_ROBUSTNESS <= 1
        ), "WEIGHT_ROBUSTNESS is greater 1 or lower 0. This is not allowed!"

        ws = 1 - wr
        fit = (abs(r) * wr + s * ws) / (abs(r_base) * wr + s_base * ws)

        if log_result:
            Result.results.append(Result(rsv=fit, s=s, r=r, s_base=s_base, r_base=r_base))
        return fit

    def eval(self, candidate_job_dict, start_times=None, log_result=True):
        """evaluate the manipulated job durations (see job_dict) regarding to its robustness and stability.
        compare it with the baselien schedule."""

        if start_times != None:
            bs = simulate_deterministic_bs_schedule(candidate_job_dict, start_times).kpis
            res = sim.run_monte_carlo_experiments(bs[hp.SCHED_OBJECTIVE], hp.SCHED_OBJECTIVE, candidate_job_dict, start_times, n_experiments=hp.N_MONTE_CARLO_EXPERIMENTS)
        else:
            bs = simulate_deterministic_bs_schedule(candidate_job_dict, self.initial_start_times).kpis
            res = sim.run_monte_carlo_experiments(
                bs[hp.SCHED_OBJECTIVE], hp.SCHED_OBJECTIVE, candidate_job_dict, self.initial_start_times, n_experiments=hp.N_MONTE_CARLO_EXPERIMENTS
            )
        fit = RobustnessEvaluator.calc_fittness(res["r_mean"], res["scom_mean"], self.initial_robustness, self.initial_stability, log_result=log_result)
        return fit, res


class BaselineSchedule:
    def __init__(self, jobs_raw):
        d.JobFactory.preprocess_jobs(jobs_raw)
        self.jobs_raw = jobs_raw
        self.job_dict = milp.get_job_dict(jobs_raw)
        start_times, _ = milp.solve(hp.SCHED_OBJECTIVE, jobs_raw)
        self.task_order = list(map(lambda x: x[0], sorted(list(start_times.items()), key=lambda x: x[1])))
        objective_value = simulate_deterministic_bs_schedule(self.job_dict, start_times).kpis[hp.SCHED_OBJECTIVE]
        self.mc_stats = get_initial_stochastic_infos(objective_value, self.job_dict, start_times)
        self.evaluator = RobustnessEvaluator(jobs_raw, start_times, self.mc_stats, objective_value)
