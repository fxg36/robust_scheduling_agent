import flowshop_milp as milp

SCHED_OBJECTIVE = None #milp.Objective.CMAX # objective for the baseline schedule
WEIGHT_ROBUSTNESS = None # weight of stability is 1-weight_robustness

N_MONTE_CARLO_EXPERIMENTS = 0 # no. of monte carlo simulations to evaluate the schedule regarding to its robustness and stability (888)

N_SAMPLES = 10 # no of samples for training/testing (see pickle file in project dir)
SAMPLES_TO_LOAD = None # 4 | 5 | 6 | 8 | 10 | 0

INCLUDE_DEMAND_RELATED_EVENTS = False # if false, only resource depended events are fire (DEPRECATED)
TENSORBOARD_LOG_PATH = "./tensorboard_log/"

CPU_CORES = 4
PROCESS_SPAWNER = None

SIM_TIMEOUT = 1