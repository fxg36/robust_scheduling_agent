import flowshop_milp as milp

SCHED_OBJECTIVE = milp.Objective.CMAX # objective for the baseline schedule
WEIGHT_ROBUSTNESS = 0.5 # weight of stability is 1-weight_robustness

NO_MONTE_CARLO_EXPERIMENTS = 3600 # no. of monte carlo simulations to evaluate the schedule regarding to its robustness and stability (888)

N_SAMPLES = 10 # no of samples for training/testing (see pickle file in project dir)
N_JOBS = 4 # no jobs for training/testing (see pickle file in project dir) [4 | 6 | 8 | 0=all jobs]
SAMPLES_TO_LOAD = 4 # 4 | 6 | 8 | 0=all jobs

INCLUDE_DEMAND_RELATED_EVENTS = False # if false, only resource depended events are fire (DEPRECATED)
TENSORBOARD_LOG_PATH = "./tensorboard_log/"

CPU_CORES = 4
PROCESS_SPAWNER = None