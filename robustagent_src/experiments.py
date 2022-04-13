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
    sa_type = 0 # 0 = add slack times, 1 = create neighbour
    sa.sa_method(sa_type, do_print=True)

def train_drl():
    hp.SAMPLES_TO_LOAD = 6
    hp.SCHED_OBJECTIVE = milp.Objective.CMAX
    hp.WEIGHT_ROBUSTNESS = 0.2
    hp.N_MONTE_CARLO_EXPERIMENTS = 600
    #ppo.train(lr_start=0.0001, gamma=0.99, training_steps=9000, steps_per_update=18)
    #a2c.train(lr_start=0.001, gamma=0.99, training_steps=9000, steps_per_update=18)

    hp.SCHED_OBJECTIVE = milp.Objective.F
    hp.N_MONTE_CARLO_EXPERIMENTS = 600
    hp.WEIGHT_ROBUSTNESS = 0.5
    ppo.train(lr_start=0.0001, gamma=0.99, training_steps=9000, steps_per_update=18)
    #a2c.train(lr_start=0.001, gamma=0.99, training_steps=9000, steps_per_update=18) 

def test_drl():
    hp.SAMPLES_TO_LOAD = 8
    hp.SCHED_OBJECTIVE = milp.Objective.F
    hp.WEIGHT_ROBUSTNESS = 0.5
    hp.N_MONTE_CARLO_EXPERIMENTS = 600
    ppo.test(test_episodes=50, result_suffix='4', sample_ids=None)


def enable_multiproc():
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING")
    #freeze_support()
    ps = ProcessSpawner(func_target=sim.experiment_process, parallel_processes=hp.CPU_CORES)
    ProcessSpawner.instances["montecarlo"] = ps
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING COMPLETED")

if __name__ == '__main__':
    enable_multiproc()
    train_drl()
    #test_drl()
    #sars1_ms()
    ProcessSpawner.instances["montecarlo"].kill_processes()