import flowshop_milp as milp
import flowshhop_simulation as sim
import hyperparam as hp


class RobustnessEvaluator:
    def __init__(self, jobs_raw):
        self.jobs = milp.get_job_dict(jobs_raw)
        assert (
            hp.WEIGHT_ROBUSTNESS >= 0 and hp.WEIGHT_ROBUSTNESS <= 1
        ), "WEIGHT_ROBUSTNESS is greater 1 or lower 0. This is not allowed!"

        # now calculate initial global optimum for the initial non-robust baseline schedule and get robustness and stability
        self.initial_start_times, self.initial_objective_value = milp.solve(hp.SCHED_OBJECTIVE, jobs_raw)
        self.initial_robustness, self.initial_stability, self.initial_stats = self._get_robustness_stability(
            self.initial_objective_value, self.jobs
        )
        self.too_conservative_schedule = self.initial_stats["r_mean"] > 0

    def _get_robustness_stability(self, objective_value, job_dict):
        res = sim.run_monte_carlo_experiments(
            objective_value,
            job_dict,
            self.initial_start_times
        )
        return abs(res["r_mean"]), abs(res["scom_mean"]), res

    def eval(self, candidate_job_dict, n_evals=1):
        """evaluate the manipulated job durations (see job_dict) regarding to its robustness and stability.
        compare it with the baselien schedule."""

        fs = sim.FlowshopSimulation(
            candidate_job_dict, self.initial_start_times, fire_dynamic_events=False
        )
        fs.env.run(until=fs.meta_proc)

        objective_value = fs.kpis[hp.SCHED_OBJECTIVE]
        #print(objective_value)

        for _ in range(n_evals):
            robustness_candidate, stability_candidate, stats = self._get_robustness_stability(
                objective_value, candidate_job_dict
            )
            if n_evals > 1:
                print(f'{robustness_candidate}/{stability_candidate}')

        robustness_rel = robustness_candidate / self.initial_robustness * hp.WEIGHT_ROBUSTNESS
        stability_rel = stability_candidate / self.initial_stability * (1 - hp.WEIGHT_ROBUSTNESS)
        eval_value = robustness_rel + stability_rel

        return eval_value, stats
