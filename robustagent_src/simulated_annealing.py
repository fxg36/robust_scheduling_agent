from copy import deepcopy
from random import choice, random, randint, seed
import flowshop_milp as milp
from robustness_evaluator import RobustnessEvaluator
import data as d
from math import exp
import hyperparam as hp

NO_ITERATIONS = 50
NO_MAX_ATTEMPTS = 5


def create_neighbor(curr_solution, robustness, temp):
    n = deepcopy(list(curr_solution.values()))
    t = choice(list(filter(lambda x: isinstance(x, list), choice(n))))
    if randint(0,1) > 0:
        t[1] *= 1 + max(0.015, 0.06 * temp)
    else:
        t[1] *= 1 - max(0.015, 0.06 * temp)
    return milp.get_job_dict(n)


def sa_method(
    jobs_raw,
    do_print=False,
):
    d.JobFactory.preprocess_jobs(jobs_raw)
    ev = RobustnessEvaluator(jobs_raw)
    best = ev.jobs
    best_eval = (1,ev.initial_stats)
    curr, curr_eval = best, best_eval
    improvements_missed = 0

    if do_print:
        print(
            f"Start with {best_eval[0]}. R={best_eval[1]['r_mean']}, S={best_eval[1]['scom_mean']} (SDur={best_eval[1]['sdur_mean']})"
        )

    for i in range(NO_ITERATIONS):
        t = (NO_ITERATIONS - i) / NO_ITERATIONS
        candidate = create_neighbor(curr, curr_eval[1]["r_mean"], t)
        candidate_eval = ev.eval(candidate, n_evals=1)

        # candidate = curr
        # candidate_eval = ev.eval(curr, n_evals=200)

        print(f"Iteration:{i}: {candidate_eval[0]}")
        if candidate_eval[0] < best_eval[0]:
            if do_print:
                print(
                    f"\tImproved from {best_eval[0]:.3f} -> {candidate_eval[0]:.3f}. R={candidate_eval[1]['r_mean']}, S={candidate_eval[1]['scom_mean']} (SDur={best_eval[1]['sdur_mean']})"
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
    jobs_p1 = randint(0, hp.NO_JOBS)
    jobs_p2 = hp.NO_JOBS - jobs_p1
    jobs_raw = d.JobFactory.get_random_jobs(jobs_p1, jobs_p2)
    sa_method(jobs_raw, do_print=True)
