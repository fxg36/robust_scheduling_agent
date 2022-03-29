# Job Shop Scheduling Problem nach Manne
import pulp
import schedule_plotter as p


class Objective:
    CMAX = "min_makespan"
    T = "min_tardiness"
    F = "min_flowtime"


def _get_x_value(x):
    """ helper to extract operation starting times from solver result"""
    assert x.name.startswith("x_"), "this must be a x variable"
    name = x.name.split("_")
    job = int(name[1])
    operation = int(name[2])
    start_time = x.varValue
    return ((job, operation), start_time)


def get_job_dict(jobs_raw):
    """ helper to convert job array to job dict """
    return {i + 1: jobs_raw[i] for i in range(len(jobs_raw))}

def solve(objective: Objective, input):
    model = pulp.LpProblem(objective, pulp.LpMinimize)

    """
    CONSTANTS
    """
    no_jobs = len(input)
    no_machines = len(input[0]) - 3
    job_dict = get_job_dict(input)
    processing_time = {(i + 1, input[i][k][0]): input[i][k][1] for i in range(no_jobs) for k in range(no_machines)}
    due_date = {i + 1: input[i][3] for i in range(no_jobs)}
    weights = {i + 1: input[i][4] for i in range(no_jobs)}

    """
    VARIABLES
    """
    # x_i,k = starting time of J_i on M_k
    x = pulp.LpVariable.dicts(
        "x", indexs=(range(1, no_jobs + 1), range(1, no_machines + 1)), cat=pulp.LpInteger, lowBound=0
    )

    # y_i,j,k = 1 if J_i precedes J_j on M_k; 0 otherwise
    y = pulp.LpVariable.dicts(
        "y", indexs=(range(1, no_jobs + 1), range(no_jobs + 1), range(1, no_machines + 1)), cat=pulp.LpBinary
    )

    """
    OBJECTIVE
    """
    if objective == Objective.CMAX:
        cmax = pulp.LpVariable(objective, lowBound=0, cat=pulp.LpInteger)
        model += cmax
        for i in range(1, no_jobs + 1):
            model += x[i][no_machines] + processing_time[i, no_machines] <= cmax

    elif objective == Objective.T:
        tmax = pulp.LpVariable(objective, lowBound=0, cat=pulp.LpInteger)
        model += tmax
        t = pulp.LpVariable.dicts("t", indexs=(range(1, no_jobs + 1)), lowBound=0)
        model += pulp.lpSum([t[j]*weights[j] for j in range(1, no_jobs + 1)]) <= tmax
        for j in range(1, no_jobs + 1):
            model += t[j] >= x[j][no_machines] + processing_time[j, no_machines] - due_date[j]
            model += t[j] >= 0

    elif objective == Objective.F:
        ft = pulp.LpVariable(objective, lowBound=0, cat=pulp.LpInteger)
        model += ft
        model += pulp.lpSum([x[i][no_machines] + processing_time[i, no_machines] for i in range(1, no_jobs + 1)]) <= ft

    """
    CONSTRAINTS
    """
    # successor operation can start, when predecessor is finished
    for i in range(1, no_jobs + 1):
        for l in range(1, no_machines):
            model += x[i][l] + processing_time[i, l] <= x[i][l+1]

    # one operation per time per machine
    M = 20000
    for j in range(1, no_jobs + 1):
        for i in range(1, no_jobs + 1):
            if i == j:
                continue
            for k in range(1, no_machines + 1):
                model += (M + processing_time[j, k]) * y[i][j][k] + (x[i][k] - x[j][k]) >= processing_time[j, k]
                model += (M + processing_time[i, k]) * (1 - y[i][j][k]) + (x[j][k] - x[i][k]) >= processing_time[i, k]


    solver = pulp.getSolver('PULP_CBC_CMD', timeLimit=120, msg=1)
    model.solve(solver)

    start_times = list(map(lambda y: _get_x_value(y), filter(lambda x: x.name.startswith("x_"), model.variables())))
    obj_value = next(filter(lambda x: x.name == objective, model.variables())).varValue

    return {s[0]: s[1] for s in start_times}, obj_value


def plot_results(jobs, start_times):
    logs = []
    for k, v in jobs.items():
        for o in range(0, 3):
            machine = v[o][0]
            start = start_times[k, o + 1]
            duration = v[o][1]
            logs.append(p.AllocationLog(k, machine, start, duration))

    p.plot("schedule planned", logs)
