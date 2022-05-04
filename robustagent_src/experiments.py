import hyperparam as hp
from process_spawner import ProcessSpawner
import rl_agent_a2c as a2c
import rl_agent_ppo as ppo
import flowshop_milp as milp
import flowshhop_simulation as sim
import simulated_annealing as sa
from robustness_evaluator import Result
import rl_agent_base as base
from copy import deepcopy
from data import JobFactory
import pandas as pd
from pathlib import Path


def sars():
    hp.WEIGHT_ROBUSTNESS = 0.5
    hp.N_MONTE_CARLO_EXPERIMENTS = 10000

    def sa_loop(sa_type):
        Result.results = []
        for i in range(10):
            sa.sa_method(sa_type, sample_id=i, do_print=True)
        Result.write_results(f"SARS{sa_type+1}_{hp.SCHED_OBJECTIVE}", hp.SAMPLES_TO_LOAD)

    for sa_type in [0, 1]:  # 0 = add slack times, 1 = create neighbour
        for samples_to_load in [5, 10, 0]:
            for obj in [milp.Objective.F, milp.Objective.CMAX]:
                hp.SCHED_OBJECTIVE = obj
                hp.SAMPLES_TO_LOAD = samples_to_load
                sa_loop(sa_type)


def train_drl():
    def train_loop(n_iterations, callback, lr_start, gamma, training_steps, steps_per_update):
        for i in range(n_iterations):
            callback(
                lr_start=lr_start,
                gamma=gamma,
                training_steps=training_steps,
                steps_per_update=steps_per_update,
                model_no=i,
            )

    models_to_train = 6
    hp.SAMPLES_TO_LOAD = 0
    hp.SCHED_OBJECTIVE = milp.Objective.CMAX
    hp.WEIGHT_ROBUSTNESS = 0.5
    hp.N_MONTE_CARLO_EXPERIMENTS = 500
    train_loop(models_to_train, ppo.train, lr_start=0.0001, gamma=0.99, training_steps=10000, steps_per_update=22)
    train_loop(models_to_train, a2c.train, lr_start=0.001, gamma=0.99, training_steps=10000, steps_per_update=22)

    hp.SCHED_OBJECTIVE = milp.Objective.F
    hp.N_MONTE_CARLO_EXPERIMENTS = 500
    hp.WEIGHT_ROBUSTNESS = 0.5
    train_loop(models_to_train, ppo.train, lr_start=0.0001, gamma=0.99, training_steps=10000, steps_per_update=22)
    train_loop(models_to_train, a2c.train, lr_start=0.001, gamma=0.99, training_steps=10000, steps_per_update=22)


def test_drl():
    hp.WEIGHT_ROBUSTNESS = 0.5
    hp.N_MONTE_CARLO_EXPERIMENTS = 10000

    best_ppo_model_flowtime = 0
    best_a2c_model_flowtime = 0
    best_ppo_model_makespan = 0
    best_a2c_model_makespan = 0

    for samples_to_load in [0, 5, 10]:
        for obj_models in [
            (milp.Objective.F, best_ppo_model_flowtime, best_a2c_model_flowtime),
            (milp.Objective.CMAX, best_ppo_model_makespan, best_a2c_model_makespan),
        ]:
            hp.SAMPLES_TO_LOAD = samples_to_load
            hp.SCHED_OBJECTIVE = obj_models[0]
            ppo.test(test_episodes=100, result_suffix=str(hp.SAMPLES_TO_LOAD), model_no=obj_models[1])
            a2c.test(test_episodes=100, result_suffix=str(hp.SAMPLES_TO_LOAD), model_no=obj_models[2])


def expected_value_tests():
    hp.N_MONTE_CARLO_EXPERIMENTS = 1000
    hp.WEIGHT_ROBUSTNESS = 1
    i = 1

    for expected_value_idx in [0, 1, 2, 3, 4]:  # see JobFactory.preprocess_one_operation
        
        def append_res(s_id, obj, jobs_raw_clone):
            JobFactory.preprocess_jobs(jobs_raw_clone, expected_value_idx=expected_value_idx)
            jobs_dict = milp.get_job_dict(jobs_raw_clone)
            _, res, _ = sample.evaluator.eval(jobs_dict, log_result=False)
            w = hp.WEIGHT_ROBUSTNESS
            results["r"].append(res["r_mean"])
            results["s"].append(res["scom_mean"])
            results["sum"].append(res["scom_mean"] + abs(res["r_mean"]))
            results["sumw"].append(res["scom_mean"] * (1 - w) + abs(res["r_mean"] * w))
            results["w"].append(w)
            results["obj"].append(obj)
            results["s_id"].append(s_id)

        for obj in [milp.Objective.F, milp.Objective.CMAX]:
            results = {"r": [], "s": [], "sum": [], "sumw": [], "obj": [], "w": [], "s_id": []}
            hp.SCHED_OBJECTIVE = obj
            samples = base.load_samples(5)
            for sample in samples:
                s_id = samples.index(sample)
                jobs_raw_clone = deepcopy(sample.jobs_raw)
                append_res(s_id, obj, jobs_raw_clone)
                print(f"{i}/{10*len(samples)} EV experiments completed")
                i += 1

            df = pd.DataFrame(results)
            df = df[["r", "s", "sum", "sumw", "w", "obj", "s_id"]]

            main_path = Path(".") / "experiment_results" / f"expected_value_tests_5jobs_{hp.SCHED_OBJECTIVE}_idx{expected_value_idx}.csv"
            df.to_csv(main_path, sep=";", encoding="utf-8", header=True, index=False, decimal=",")
            descr = df.describe()
            stats_path = Path(".") / "experiment_results" / f"expected_value_tests_5jobs_{hp.SCHED_OBJECTIVE}_idx{expected_value_idx}_stats.csv"
            descr.to_csv(stats_path, sep=";", encoding="utf-8", header=True, index=True, decimal=",")
            print(df)
            print(descr)




def enable_multiproc():
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING")
    ps = ProcessSpawner(func_target=sim.experiment_process, parallel_processes=hp.CPU_CORES)
    ProcessSpawner.instances["montecarlo"] = ps
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING COMPLETED")


if __name__ == "__main__":

    # X = {}
    # X[1,1] = 0
    # X[1,2] = 50
    # X[2,1] = 65
    # X[2,2] = 80
    # X[3,1] = 60
    # X[3,2] = 120

    # S = list(map(lambda x: (x[0],x[1]), sorted(list(X.items()), key=lambda x: (x[0][1],x[0][0]))))

    enable_multiproc()
    #expected_value_tests()
    train_drl()
    # sars()
    # test_drl()
    ProcessSpawner.instances["montecarlo"].kill_processes()