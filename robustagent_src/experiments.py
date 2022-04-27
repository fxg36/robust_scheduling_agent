import hyperparam as hp
from process_spawner import ProcessSpawner
import rl_agent_a2c as a2c
import rl_agent_ppo as ppo
import flowshop_milp as milp
import flowshhop_simulation as sim
import simulated_annealing as sa


def sars1_ms():
    hp.SCHED_OBJECTIVE = milp.Objective.F
    hp.WEIGHT_ROBUSTNESS = 0.85
    sa_type = 0  # 0 = add slack times, 1 = create neighbour
    sa.sa_method(sa_type, do_print=True)


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

    hp.SAMPLES_TO_LOAD = 0
    hp.SCHED_OBJECTIVE = milp.Objective.CMAX
    hp.WEIGHT_ROBUSTNESS = 0.5
    hp.N_MONTE_CARLO_EXPERIMENTS = 400
    # ppo.train(lr_start=0.0001, gamma=0.99, training_steps=9000, steps_per_update=18)
    # a2c.train(lr_start=0.001, gamma=0.99, training_steps=9000, steps_per_update=18)
    #train_loop(5, ppo.train, lr_start=0.0001, gamma=0.99, training_steps=10000, steps_per_update=18)
    #train_loop(5, a2c.train, lr_start=0.001, gamma=0.99, training_steps=10000, steps_per_update=18)


    hp.SCHED_OBJECTIVE = milp.Objective.F
    hp.N_MONTE_CARLO_EXPERIMENTS = 400
    hp.WEIGHT_ROBUSTNESS = 0.5
    # train_loop(5, ppo.train, lr_start=0.0001, gamma=0.99, training_steps=10000, steps_per_update=18)
    # train_loop(5, a2c.train, lr_start=0.001, gamma=0.99, training_steps=10000, steps_per_update=18)
    # ppo.train(lr_start=0.0001, gamma=0.99, training_steps=10000, steps_per_update=10)
    # a2c.train(lr_start=0.001, gamma=0.99, training_steps=9000, steps_per_update=18)


def test_drl():
    hp.WEIGHT_ROBUSTNESS = 0.5
    hp.N_MONTE_CARLO_EXPERIMENTS = 400
    hp.SCHED_OBJECTIVE = milp.Objective.F
    hp.SAMPLES_TO_LOAD = 4
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=4)
    hp.SAMPLES_TO_LOAD = 6
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=4)
    hp.SAMPLES_TO_LOAD = 8
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=4)
    hp.SAMPLES_TO_LOAD = 0
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=4)

    hp.SCHED_OBJECTIVE = milp.Objective.CMAX
    hp.SAMPLES_TO_LOAD = 4
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=0)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)
    hp.SAMPLES_TO_LOAD = 6
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=0)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)
    hp.SAMPLES_TO_LOAD = 8
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=0)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)
    hp.SAMPLES_TO_LOAD = 0
    ppo.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=0)
    a2c.test(test_episodes=50, result_suffix=str(hp.SAMPLES_TO_LOAD), sample_ids=None, model_no=2)


def enable_multiproc():
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING")
    # freeze_support()
    ps = ProcessSpawner(func_target=sim.experiment_process, parallel_processes=hp.CPU_CORES)
    ProcessSpawner.instances["montecarlo"] = ps
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING COMPLETED")


if __name__ == "__main__":
    enable_multiproc()
    #train_drl()
    test_drl()
    # sars1_ms()
    ProcessSpawner.instances["montecarlo"].kill_processes()