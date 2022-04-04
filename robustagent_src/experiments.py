from multiprocessing import freeze_support
import hyperparam as hp
from process_spawner import ProcessSpawner
import rl_agent_a2c as a2c
import rl_agent_ppo as ppo
import flowshop_milp as milp
import flowshhop_simulation as sim
import simulated_annealing as sa

def sars1_ms():
    sa_type = 0 # 0 = add slack times, 1 = create neighbour
    sa.sa_method(sa_type, do_print=True)

def train_ms():
    hp.N_JOBS = 4
    hp.SCHED_OBJECTIVE = milp.Objective.CMAX
    ppo.train(lr_start=0.0004, gamma=0.96, training_steps=6000, steps_per_update=12)
    #a2c.train(lr_start=0.0005, gamma=0.98, training_steps=60000, steps_per_update=10)

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
    train_ms()
    #sars1_ms()