from copy import deepcopy
from random import choice, random, randint, sample
import flowshop_milp as milp
from math import exp
import hyperparam as hp
import rl_agent_base as base

NO_ITERATIONS = 50
NO_MAX_ATTEMPTS = 5


def create_neighbor(curr_solution, temp, sa_type):
    n = deepcopy(curr_solution)
    if sa_type == 0:
        jobs = list(n[0].values())
        t = choice(list(filter(lambda x: isinstance(x, list), choice(jobs))))
        t[1] *= 1 + max(0.015, 0.06 * temp)
        if randint(0, 1) > 0:
            t[1] *= 1 + max(0.02, 0.1 * temp)
        else:
            t[1] *= 1 - max(0.02, 0.1 * temp)
    elif sa_type == 1:
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
    do_print=False,
):
    samples = base.load_samples()
    bs: base.BaselineSchedule
    bs = samples[3]
    ev = bs.evaluator
    best = (ev.job_dict, ev.initial_start_times)
    best_eval = (
        ev.calc_fittness(
            ev.initial_stats["r_mean"],
            ev.initial_stats["scom_mean"],
            ev.initial_stats["r_mean"],
            ev.initial_stats["scom_mean"]
        ),
        ev.initial_stats,
    )
    curr, curr_eval = best, best_eval
    improvements_missed = 0

    if do_print:
        print(
            f"Start with {best_eval[0]}. R={best_eval[1]['r_mean']}, S={best_eval[1]['scom_mean']} (SDur={best_eval[1]['sdur_mean']})"
        )

    for i in range(NO_ITERATIONS):
        t = (NO_ITERATIONS - i) / NO_ITERATIONS
        # candidate = create_neighbor(curr, curr_eval[1]["r_mean"], t)
        candidate = create_neighbor(curr, t, sa_type)
        if sa_type == 1:
            candidate_eval = ev.eval(candidate[0], candidate[1], neighbor_robustness=0, neighbor_stability=0)
        else:
            candidate_eval = ev.eval(candidate[0], candidate[1])

        print(f"Iteration:{i}: {candidate_eval[0]}")
        if candidate_eval[0] < best_eval[0]:
            if do_print:
                obj = candidate_eval[1]['makespan_mean'] if hp.SCHED_OBJECTIVE == milp.Objective.CMAX else candidate_eval[1]['flowtime_mean']
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

    return best, best_eval


if __name__ == "__main__":
    sa_type = 1 # 0 = add slack times, 1 = create neighbour
    sa_method(sa_type, do_print=True)
