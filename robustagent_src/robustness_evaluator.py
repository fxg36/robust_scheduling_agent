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
    return fs.kpis

def get_initial_stochastic_infos(obj_value, job_dict, start_times):
    return sim.run_monte_carlo_experiments(
            obj_value, job_dict, start_times, n_experiments=10000
        )

class Result:
    results = []
    def __init__(self, rsv, s, r, s_base, r_base):
        self.rsv = rsv
        self.s = s
        self.r = r
        self.s_base = s_base
        self.r_base = r_base
    

    @staticmethod
    def write_results(filename, suffix, reset=True):
        p = Path(".")
        p = p / "experiment_results" / f"{filename}_{suffix}.csv"

        # for v in Result.results:
        #     print(v)
        # print(f"min: {min(V)}")
        # print(f"mean: {np.mean(V)}")
        # print(f"std: {np.std(V)}")

        #data = list(map(lambda x: [x.rsv, x.r, x.s, x.r_base, x.s_base], Result.results))

        df = pd.DataFrame({
            'rsv':list(map(lambda x: x.rsv, Result.results)),
            'r':list(map(lambda x: x.r, Result.results)),
            'r_base':list(map(lambda x: x.r_base, Result.results)),
            'r_diff': list(map(lambda x: abs(x.r_base)-abs(x.r), Result.results)),
            's':list(map(lambda x: x.s, Result.results)),
            's_base':list(map(lambda x: x.s_base, Result.results)),
            's_diff': list(map(lambda x: x.s_base-x.s, Result.results)),
        })
        df = df[['rsv', 'r', 'r_base', 'r_diff', 's', 's_base', 's_diff']] 
        df.to_csv(p, sep=';', encoding='utf-8', header=True, index=False, decimal=",")
        descr = df.describe()
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
    def calc_fittness(r, s, r_base, s_base):
        wr = hp.WEIGHT_ROBUSTNESS
        ws = 1 - wr
        fit = (abs(r) * wr + s * ws) / (abs(r_base) * wr + s_base * ws)
        Result.results.append(Result(rsv=fit, s=s, r=r, s_base=s_base, r_base=r_base))
        return fit
        # robustness_rel = (abs(r)) / (abs(r_base)) * hp.WEIGHT_ROBUSTNESS
        # stability_rel = (s) / (s_base) * (1 - hp.WEIGHT_ROBUSTNESS)
        # eval_value = robustness_rel + stability_rel
        # return eval_value

    def eval(self, candidate_job_dict, start_times=None):
        """evaluate the manipulated job durations (see job_dict) regarding to its robustness and stability.
        compare it with the baselien schedule."""

        if start_times != None:
            bs = simulate_deterministic_bs_schedule(candidate_job_dict, start_times)
            res = sim.run_monte_carlo_experiments(bs[hp.SCHED_OBJECTIVE], candidate_job_dict, start_times)
        else:
            bs = simulate_deterministic_bs_schedule(candidate_job_dict, self.initial_start_times)
            #assert bs[hp.SCHED_OBJECTIVE] == self.initial_objective_value, 'must be true if all actions are [0]'
            res = sim.run_monte_carlo_experiments(
                bs[hp.SCHED_OBJECTIVE], candidate_job_dict, self.initial_start_times
            )
        fit = RobustnessEvaluator.calc_fittness(res["r_mean"], res["scom_mean"], self.initial_robustness, self.initial_stability)
        return fit, res


class BaselineSchedule:
    def __init__(self, jobs_raw):
        d.JobFactory.preprocess_jobs(jobs_raw)
        self.jobs_raw = jobs_raw
        self.job_dict = milp.get_job_dict(jobs_raw)
        start_times, _ = milp.solve(hp.SCHED_OBJECTIVE, jobs_raw)
        self.task_order = list(map(lambda x: x[0], sorted(list(start_times.items()), key=lambda x: x[1])))
        objective_value = simulate_deterministic_bs_schedule(self.job_dict, start_times)[hp.SCHED_OBJECTIVE]
        self.mc_stats = get_initial_stochastic_infos(objective_value, self.job_dict, start_times)
        self.evaluator = RobustnessEvaluator(jobs_raw, start_times, self.mc_stats, objective_value)
        #assert hp.N_JOBS != 0 and len(jobs_raw) == hp.N_JOBS, "NO JOBS is not set properly!"
