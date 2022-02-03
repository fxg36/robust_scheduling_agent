from typing import List
from schedule_plotter import AllocationLog


def _get_successors(l: AllocationLog, logs: List[AllocationLog]):
    """ get the next two direct succeeding operations. (one job succeedor and one machine succeedor) """
    operation_succ = list(filter(lambda x: x.job_no == l.job_no and x.task_start >= l.task_end, logs))
    machine_succ = list(filter(lambda x: x.machine_no == l.machine_no and x.task_start >= l.task_end, logs))
    operation_succ = sorted(operation_succ, key=lambda x: x.task_start)
    machine_succ = sorted(machine_succ, key=lambda x: x.task_start)
    res = []
    if operation_succ:
        res.append(operation_succ[0])
    if machine_succ:
        res.append(machine_succ[0])
    return res


def _get_predecessors(l: AllocationLog, logs: List[AllocationLog]):
    """ get the next two direct preceeding operations. (one job preceedor and one machine preceedor) """
    operation_pre = list(filter(lambda x: x.job_no == l.job_no and x.task_end <= l.task_start, logs))
    machine_pre = list(filter(lambda x: x.machine_no == l.machine_no and x.task_end <= l.task_start, logs))
    operation_pre = sorted(operation_pre, key=lambda x: x.task_end, reverse=True)
    machine_pre = sorted(machine_pre, key=lambda x: x.task_end, reverse=True)
    res = []
    if operation_pre:
        res.append(operation_pre[0])
    if machine_pre:
        res.append(machine_pre[0])
    return res


def calc_slack(logs: List[AllocationLog]):
    """ calculates free slack and total slack with standard pdm method """
    es = {}  # earliest start
    ee = {}  # earliest end
    ls = {}  # latest start
    le = {}  # latest end

    first = sorted(logs, key=lambda x: x.task_start)[0]
    c = first # current
    es[c.job_no, c.machine_no] = 0
    ee[c.job_no, c.machine_no] = c.task_end
    nexts = []
    while True:
        if c != first:
                pres = _get_predecessors(c, logs)
                es[c.job_no, c.machine_no] = max(list(map(lambda x: ee[x.job_no, x.machine_no], pres)))
                ee[c.job_no, c.machine_no] = c.task_duration + es[c.job_no, c.machine_no]
        for s in _get_successors(c, logs):
            pres = set(_get_predecessors(s, logs))
            if set(map(lambda x: (x.job_no, x.machine_no), pres)).issubset(set(es.keys())) and s not in nexts:
                nexts.append(s)
        assert len(ee.keys()) < len(logs) and any(nexts) or len(ee.keys()) == len(logs) and not any(nexts), 'this must be true!'
        if len(ee.keys()) == len(logs):
            break
        c = nexts.pop(0)

    last = sorted(logs, key=lambda x: x.task_end, reverse=True)[0]
    c = last
    le[c.job_no, c.machine_no] = ee[c.job_no, c.machine_no]
    ls[c.job_no, c.machine_no] = le[c.job_no, c.machine_no] - c.task_duration
    nexts = []
    while True:
        if c != last:
            le[c.job_no, c.machine_no] = min(list(map(lambda x: ls[x.job_no, x.machine_no], _get_successors(c, logs))))
            ls[c.job_no, c.machine_no] = le[c.job_no, c.machine_no] - c.task_duration
        for p in _get_predecessors(c, logs):
            succs = set(_get_successors(p, logs))
            if set(map(lambda x: (x.job_no, x.machine_no), succs)).issubset(set(ls.keys())) and p not in nexts:
                nexts.append(p)
        assert len(le.keys()) < len(logs) and any(nexts) or len(le.keys()) == len(logs) and not any(nexts), 'this must be true!'
        if len(le.keys()) == len(logs):
            break
        c = nexts.pop(0)

    total_slack = {}
    for l in logs:
        total_slack[l.job_no, l.machine_no] = ls[l.job_no, l.machine_no] - es[l.job_no, l.machine_no]

    free_slack = {}
    for l in logs:
        succ_ess = []
        for s in _get_successors(l, logs):
            succ_ess.append(es[s.job_no, s.machine_no])
        free_slack[l.job_no, l.machine_no] = 0 if not succ_ess else min(succ_ess) - ee[l.job_no, l.machine_no]

    return {"total_slack": total_slack, "free_slack": free_slack}