import flowshop_milp as milp

SCHED_OBJECTIVE = milp.Objective.CMAX # objective for the baseline schedule
WEIGHT_ROBUSTNESS = 0.5 # weight of stability is 1-weight_robustness
NO_JOBS = 4 # no jobs for training/testing
NO_MONTE_CARLO_EXPERIMENTS = 200 # no. of monte carlo simulations to evaluate the schedule regarding to its robustness and stability
INCLUDE_DEMAND_RELATED_EVENTS = False # if false, only resource depended events are fire
TENSORBOARD_LOG_PATH = "./tensorboard_log/"