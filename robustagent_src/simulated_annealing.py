from copy import deepcopy
from random import choice, random, randint, sample
import flowshop_milp as milp
from math import exp
import hyperparam as hp
from process_spawner import ProcessSpawner
import rl_agent_base as base
from robustness_evaluator import RobustnessEvaluator
from data import JobFactory as jf
import flowshhop_simulation as sim
from robustness_evaluator import Result
import time

NO_ITERATIONS = 50
NO_MAX_ATTEMPTS = 5


def create_neighbor(curr_solution, temp, sa_type):
    n = deepcopy(curr_solution)
    if sa_type == 0:  # stretching/compressing operations
        jobs = list(n[0].values())
        rnd_job = choice(jobs)
        job_no = jobs.index(rnd_job) + 1
        t = choice(list(filter(lambda x: isinstance(x, list), rnd_job)))
        processing_time_candidates = jf.preprocess_one_operation(jobs_raw=jobs, job_id=job_no, machine_id=t[0])
        time_default = processing_time_candidates[2]
        time_optimistic = processing_time_candidates[1]
        time_conservative = processing_time_candidates[3]
        times = [time_default, time_optimistic, time_conservative]
        t[1] = choice(times)

    elif sa_type == 1:  # neighbor generation
        machine = randint(1, 3)
        swap = sample(list(filter(lambda x: x[1] == machine, list(n[1].keys()))), 2)
        temp = n[1][swap[0]]
        n[1][swap[0]] = n[1][swap[1]]
        n[1][swap[1]] = temp
    else:
        raise Exception("sa_type not defined")
    return n


def sa_method(
    sa_type,
    sample_id=0,
    do_print=False,
):
    start = time.time()
    samples = base.load_samples(hp.SAMPLES_TO_LOAD)

    bs: base.BaselineSchedule
    bs = samples[sample_id]
    ev = bs.evaluator
    best = (ev.job_dict, ev.initial_start_times)
    best_eval = (
        RobustnessEvaluator.calc_fittness(
            ev.initial_stats["r_mean"],
            ev.initial_stats["scom_mean"],
            ev.initial_stats["r_mean"],
            ev.initial_stats["scom_mean"],
            log_result=False,
        ),
        ev.initial_stats,
    )
    curr, curr_eval = best, best_eval
    improvements_missed = 0
    initial_objective_value = bs.evaluator.initial_objective_value

    if do_print:
        print(
            f"Start with {best_eval[0]}. R={best_eval[1]['r_mean']}, S={best_eval[1]['scom_mean']} (SDur={best_eval[1]['sdur_mean']})"
        )

    for i in range(NO_ITERATIONS):
        t = (NO_ITERATIONS - i) / NO_ITERATIONS
        candidate = create_neighbor(curr, t, sa_type)
        if sa_type == 1: # 0 = add slack times, 1 = create neighbour
            candidate_eval = ev.eval(
                candidate[0], start_times_override=candidate[1], log_result=False
            )
        else:
            candidate_eval = ev.eval(candidate[0], log_result=False)
        
        if do_print:
            print(f"Iteration:{i}: {candidate_eval[0]}")
        
        if candidate_eval[0] < best_eval[0]:
            if do_print:
                obj = (
                    candidate_eval[1]["makespan_mean"]
                    if hp.SCHED_OBJECTIVE == milp.Objective.CMAX
                    else candidate_eval[1]["flowtime_mean"]
                )
                print(
                    f"\tImproved from {best_eval[0]:.3f} -> {candidate_eval[0]:.3f}. R={candidate_eval[1]['r_mean']}, S={candidate_eval[1]['scom_mean']} (SDur={best_eval[1]['sdur_mean']} | {obj})"
                )
            best, best_eval = candidate, candidate_eval
            improvements_missed = 0
        else:
            improvements_missed += 1

        diff = candidate_eval[0] - curr_eval[0]

        if improvements_missed >= NO_MAX_ATTEMPTS:
            curr, curr_eval = best, best_eval
            improvements_missed = 0
        elif diff < 0 or random() < exp(-diff / t):
            curr, curr_eval = candidate, candidate_eval
    
    Result.results.append(
        Result(
            rsv=best_eval[0],
            r=best_eval[1]["r_mean"],
            s=best_eval[1]["scom_mean"],
            r_base=ev.initial_stats["r_mean"],
            s_base=ev.initial_stats["scom_mean"],
            time=time.time() - start,
            objective_diff = best_eval[2] / initial_objective_value if sa_type == 1 else None
        )
    )

    return best, best_eval


if __name__ == "__main__":
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING")
    # freeze_support()
    ps = ProcessSpawner(func_target=sim.experiment_process, parallel_processes=hp.CPU_CORES)
    ProcessSpawner.instances["montecarlo"] = ps
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING COMPLETED")

    hp.WEIGHT_ROBUSTNESS = 0.5
    hp.SAMPLES_TO_LOAD = 4
    hp.SCHED_OBJECTIVE = milp.Objective.CMAX
    hp.N_MONTE_CARLO_EXPERIMENTS = 4000

    sa_type = 0  # 0 = add slack times, 1 = create neighbour
    sa_method(sa_type, do_print=True)
