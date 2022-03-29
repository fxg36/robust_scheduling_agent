import flowshop_milp as milp

SCHED_OBJECTIVE = milp.Objective.F # objective for the baseline schedule
WEIGHT_ROBUSTNESS = 0.5 # weight of stability is 1-weight_robustness
SAMPLES = 5 # no of samples for training/testing (see pickle file in project dir)
NO_JOBS = 4 # no jobs for training/testing (see pickle file in project dir)
NO_MONTE_CARLO_EXPERIMENTS = 300 # no. of monte carlo simulations to evaluate the schedule regarding to its robustness and stability (888)
INCLUDE_DEMAND_RELATED_EVENTS = False # if false, only resource depended events are fire
TENSORBOARD_LOG_PATH = "./tensorboard_log/"