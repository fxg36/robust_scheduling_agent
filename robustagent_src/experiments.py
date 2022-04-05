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
    hp.WEIGHT_ROBUSTNESS = 0.15
    hp.NO_MONTE_CARLO_EXPERIMENTS = 200
    ppo.train(lr_start=0.00005, gamma=1, training_steps=6300, steps_per_update=18)
    #a2c.train(lr_start=0.001, gamma=1, training_steps=6300, steps_per_update=18)

    hp.SCHED_OBJECTIVE = milp.Objective.F
    hp.WEIGHT_ROBUSTNESS = 0.85
    #ppo.train(lr_start=0.0001, gamma=1, training_steps=6300, steps_per_update=18)
    #a2c.train(lr_start=0.001, gamma=1, training_steps=6300, steps_per_update=18) 

def test_drl():
    hp.SAMPLES_TO_LOAD = 6
    hp.SCHED_OBJECTIVE = milp.Objective.CMAX
    hp.WEIGHT_ROBUSTNESS = 0.15
    ppo.test(test_episodes=20, result_suffix='4')

# Train A2C MS model with 4 jobs
# Train A2C FT model with 4 jobs
# Train PPO MS model with 4 jobs
# Train PPO FT model with 4 jobs

# Test A2C models on 4, 6 and 8 jobs

# Test PPO models on 4, 6 and 8 jobs


# Train PPO MS model with 4,6,8 jobs

# Test PPO models on 4, 6 and 8 jobs

# pick random samples to benchmark SARS

# PPO (1 pred) - 4,6,8 jobs
# PPO (10 preds) - 4,6,8 jobs

# SARS1 (10/50) - 4,6,8 jobs
# SARS2 (10 preds) - 4,6,8 jobs

def enable_multiproc():
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING")
    #freeze_support()
    ps = ProcessSpawner(func_target=sim.experiment_process, parallel_processes=hp.CPU_CORES)
    ProcessSpawner.instances["montecarlo"] = ps
    print("INIT PROCESS SPAWNER FOR PARALLEL COMPUTING COMPLETED")

if __name__ == '__main__':
    enable_multiproc()
    #train_drl()
    test_drl()
    #sars1_ms()