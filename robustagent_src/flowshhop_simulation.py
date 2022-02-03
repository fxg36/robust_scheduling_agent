from typing import List
import simpy
from simpy.resources.store import FilterStore, Store
import schedule_plotter as p
import numpy as np
import data as d
import flowshop_milp as milp
import precedence_diagram_method
import hyperparam as hp


class SimJob:
    def __init__(
        self,
        job_no: int,
        start_times_planned,
        durations_planned,
        due_date: int,
        tardiness_weight: int,
        prod_cat,
        fire_dynamic_events=False,
    ):
        self.job_no = job_no
        self.start_times_planned = start_times_planned
        self.durations_planned = durations_planned
        self.due_date = due_date
        self.tardiness_weight = tardiness_weight
        self.prod_cat = prod_cat

        def get_duration(tup):
            return int(round(np.random.triangular(tup[0], tup[1], tup[2], 1)[0]))

        def stochastic_machine_downtime(m_no):
            if np.random.rand() <= (
                d.MachineFailure.M1FailureProb
                if m_no == 1
                else d.MachineFailure.M2FailureProb
                if m_no == 2
                else d.MachineFailure.M3FailureProb
            ):
                return get_duration(
                    d.MachineFailure.M1MaintenanceEffort
                    if m_no == 1
                    else d.MachineFailure.M2MaintenanceEffort
                    if m_no == 2
                    else d.MachineFailure.M3MaintenanceEffort
                )
            return 0

        if fire_dynamic_events:
            self.durations_actual = {
                1: get_duration(d.Product1.M1Duration if prod_cat == "p1" else d.Product2.M1Duration)
                + stochastic_machine_downtime(1),
                2: get_duration(d.Product1.M2Duration if prod_cat == "p1" else d.Product2.M2Duration)
                + stochastic_machine_downtime(2),
                3: get_duration(d.Product1.M3Duration if prod_cat == "p1" else d.Product2.M3Duration)
                + stochastic_machine_downtime(3),
            }
        else:
            self.durations_actual = durations_planned

        self.start_times_actual = {}
        self.end_time_actual = -1
        self.processing_state = 1


class FlowshopSimulation:
    def __init__(self, job_dict, start_times_planned, fire_dynamic_events=True):
        self.env = simpy.Environment()
        self.job_drain = Store(self.env)
        self.job_source = FilterStore(self.env)
        self.job_buffer = {}
        self.origin_job_dict = job_dict
        self.machine_nos = set(map(lambda x: x[0][1], start_times_planned.items()))
        for job_no in job_dict.keys():  # add planned jobs
            start_times = sorted(
                filter(lambda x: x[0][0] == job_no, start_times_planned.items()), key=lambda x: x[0][1]
            )
            self.job_source.put(
                SimJob(
                    job_no=job_no,
                    start_times_planned={s[0][1]: s[1] for s in start_times},
                    durations_planned={s[0][1]: job_dict[job_no][s[0][1] - 1][1] for s in start_times},
                    prod_cat=job_dict[job_no][5],
                    due_date=job_dict[job_no][3],
                    tardiness_weight=job_dict[job_no][4],
                    fire_dynamic_events=fire_dynamic_events,
                ),
            )
        self.meta_proc = self.env.process(self._meta_process(no_jobs=len(self.job_source.items)))
        for machine_no in self.machine_nos:
            processing_order_machine = list(
                map(
                    lambda y: y[0][0],
                    sorted(filter(lambda x: x[0][1] == machine_no, start_times_planned.items()), key=lambda x: x[1]),
                )
            )
            processing_order_machine.extend(
                map(
                    lambda x: x.job_no,
                    filter(lambda y: y.job_no not in processing_order_machine, self.job_source.items),
                )
            )
            self.job_buffer[machine_no] = FilterStore(self.env)
            self.env.process(self._machine_process(machine_no, processing_order_machine))

        self.kpis = {}

    def _update_buffers(self):
        for job in self.job_source.items:
            self.job_buffer[job.processing_state].put(self.job_source.get(lambda x: x.job_no == job.job_no).value)

    def _meta_process(self, no_jobs):
        while True:
            self._update_buffers()
            if len(self.job_drain.items) == no_jobs:  # all jobs finished
                # calculate kpis
                self.kpis[milp.Objective.CMAX] = self.env.now
                self.kpis[milp.Objective.F] = sum(map(lambda x: x.end_time_actual, self.job_drain.items))
                self.kpis[milp.Objective.T] = sum(
                    map(lambda x: max(0, x.end_time_actual - x.due_date) * x.tardiness_weight, self.job_drain.items)
                )
                self.logs = []
                for j in self.job_drain.items:
                    self.logs.extend(
                        [
                            p.AllocationLog(j.job_no, m, j.start_times_actual[m], j.durations_actual[m])
                            for m in j.start_times_actual.keys()
                        ]
                    )
                wratios = {}
                for machine_no in self.machine_nos:
                    mlogs = sorted(list(filter(lambda x: x.machine_no == machine_no, self.logs)), key=lambda x: x.task_start)
                    m_start = min(map(lambda x: x.task_start, mlogs))
                    m_end = max(map(lambda x: x.task_end, mlogs))
                    w = 0
                    for i in range(len(mlogs)-1):
                        curr_log = mlogs[i]
                        next_log = mlogs[i+1]
                        w += next_log.task_start - curr_log.task_end
                    wratios[machine_no] = w / (m_end-m_start)
                self.kpis["machine_waittime_ratio"] = wratios
                return
            yield self.env.timeout(1)

    def _machine_process(self, machine_no: int, processing_order: List[int]):
        while True:
            next_job = processing_order.pop(0)  # get next job by processing order for this machine
            job = yield self.job_buffer[machine_no].get(lambda x: x.job_no == next_job)
            job.start_times_actual[machine_no] = self.env.now
            yield self.env.timeout(job.durations_actual[machine_no])
            if machine_no == 3:  # job is finished on last machine -> put to drain
                job.end_time_actual = self.env.now
                self.job_drain.put(job)
            else:
                job.processing_state += 1
                self.job_source.put(job)
                self._update_buffers()
            if not processing_order:  # all jobs finished for this machine
                return

    def get_stability_metrices(self):
        processing_time_diffs = {}
        processing_times = {}
        completion_time_diffs = {}
        completion_times = {}
        for job_no in self.origin_job_dict.keys():
            simjob: SimJob
            simjob = next(filter(lambda x: x.job_no == job_no, self.job_drain.items))
            for machine_no, actual_duration in simjob.durations_actual.items():
                processing_times[job_no, machine_no] = actual_duration
                processing_time_diffs[job_no, machine_no] = abs(simjob.durations_planned[machine_no] - actual_duration)
                completion_times[job_no, machine_no] = simjob.start_times_actual[machine_no] + actual_duration
                completion_time_diffs[job_no, machine_no] = abs(
                    simjob.start_times_planned[machine_no]
                    + simjob.durations_planned[machine_no]
                    - completion_times[job_no, machine_no]
                )
        return (
            completion_times,
            sum(completion_time_diffs.values()),
            processing_times,
            sum(processing_time_diffs.values()),
            precedence_diagram_method.calc_slack(self.logs),
        )


def run_monte_carlo_experiments(
    objective_value,
    jobs,
    start_times,
    n_experiments = hp.NO_MONTE_CARLO_EXPERIMENTS
):
    """run several experiments to validate the schedule regarding robustness and stability.
    robustness := effectiveness regarding the objective (0: realistic predictive schedule, >0: schedule too conservative, <0: schedule too optimistic)
    stability := indicator, of how realistically the operation process times were estimated.
    (0: realistic durations, >0: schedule too conservative, <0: schedule too optimistic)
    """
    monte_carlo_results = []
    for _ in range(n_experiments):
        sim = FlowshopSimulation(jobs, start_times, fire_dynamic_events=True)
        sim.env.run(until=sim.meta_proc)
        com, stab_com_sum, dur, stab_dur_sum, slack = sim.get_stability_metrices()
        monte_carlo_results.append(
            {
                "robustness": objective_value - sim.kpis[hp.SCHED_OBJECTIVE],
                "stability_completion": stab_com_sum,
                "stability_duration": stab_dur_sum,
                "completions": com,
                "durations": dur,
                "free_slack": slack["free_slack"],
                "total_slack": slack["total_slack"],
                "makespan": sim.kpis[milp.Objective.CMAX],
                "flowtime": sim.kpis[milp.Objective.F],
                "machine_wait_time_ratio": sim.kpis['machine_waittime_ratio']
            }
        )
    mean_job_duration = {}
    std_job_duration = {}
    mean_job_completion = {}
    std_job_completion = {}
    mean_job_freeslack = {}
    std_job_freeslack = {}
    mean_job_totalslack = {}
    std_job_totalslack = {}
    comps_joined = []
    durs_joined = []
    ts_joined = []
    fs_joined = []
    [comps_joined.extend(d) for d in map(lambda x: list(x["completions"].items()), monte_carlo_results)]
    [durs_joined.extend(d) for d in map(lambda x: list(x["durations"].items()), monte_carlo_results)]
    [ts_joined.extend(d) for d in map(lambda x: list(x["free_slack"].items()), monte_carlo_results)]
    [fs_joined.extend(d) for d in map(lambda x: list(x["total_slack"].items()), monte_carlo_results)]
    
    for job_no in jobs.keys():
        dur_job = list(filter(lambda x: x[0][0] == job_no, durs_joined))
        comp_job = list(filter(lambda x: x[0][0] == job_no, comps_joined))
        ts_job = list(filter(lambda x: x[0][0] == job_no, ts_joined))
        fs_job = list(filter(lambda x: x[0][0] == job_no, fs_joined))

        for machine_no in set(map(lambda x: x[0][1], dur_job)):
            dur_machine = list(map(lambda y: y[1], filter(lambda x: x[0][1] == machine_no, dur_job)))
            comp_machine = list(map(lambda y: y[1], filter(lambda x: x[0][1] == machine_no, comp_job)))
            ts_machine = list(map(lambda y: y[1], filter(lambda x: x[0][1] == machine_no, ts_job)))
            fs_machine = list(map(lambda y: y[1], filter(lambda x: x[0][1] == machine_no, fs_job)))
            mean_job_completion[job_no, machine_no] = np.mean(comp_machine)
            mean_job_duration[job_no, machine_no] = np.mean(dur_machine)
            std_job_completion[job_no, machine_no] = np.std(comp_machine)
            std_job_duration[job_no, machine_no] = np.std(dur_machine)
            mean_job_freeslack[job_no, machine_no] = np.mean(fs_machine)
            std_job_freeslack[job_no, machine_no] = np.std(fs_machine)
            mean_job_totalslack[job_no, machine_no] = np.mean(ts_machine)
            std_job_totalslack[job_no, machine_no] = np.std(ts_machine)

    machine_wait_time_ratio_mean = {}
    for machine_no in sim.machine_nos:
        machine_wait_time_ratio_mean[machine_no] =  np.mean(list(map(lambda x: x["machine_wait_time_ratio"][machine_no], monte_carlo_results)))

    return {
        "r_mean": np.mean(list(map(lambda x: x["robustness"], monte_carlo_results))),
        "r_std": np.std(list(map(lambda x: x["robustness"], monte_carlo_results))),
        "scom_mean": np.mean(list(map(lambda x: x["stability_completion"], monte_carlo_results))),
        "sdur_mean": np.mean(list(map(lambda x: x["stability_duration"], monte_carlo_results))),
        "scom_std": np.std(list(map(lambda x: x["stability_completion"], monte_carlo_results))),
        "sdur_std": np.std(list(map(lambda x: x["stability_duration"], monte_carlo_results))),
        "dur_mean_job": mean_job_duration,
        "com_mean_job": mean_job_completion,
        "dur_std_job": std_job_duration,
        "com_std_job": std_job_completion,
        "free_slack_mean_job": mean_job_freeslack,
        "free_slack_std_job": std_job_freeslack,
        "total_slack_mean_job": mean_job_totalslack,
        "total_slack_std_job": std_job_totalslack,
        "mean_total_slack": np.mean(list(mean_job_totalslack.values())),
        "mean_free_slack": np.mean(list(mean_job_freeslack.values())),
        "makespan_mean":  np.mean(list(map(lambda x: x["makespan"], monte_carlo_results))),
        "flowtime_mean":  np.mean(list(map(lambda x: x["flowtime"], monte_carlo_results))),
        "machine_wait_time_ratio_mean":  machine_wait_time_ratio_mean
    }
