from process_spawner import ProcessSpawner
import rl_agent_base as base
import hyperparam as hp
from flowshop_milp import Objective, get_job_dict
from data import JobFactory
from copy import deepcopy
import flowshhop_simulation as sim
import pandas as pd

def append_res(s_id, obj, jobs_raw_clone):
    JobFactory.preprocess_jobs(jobs_raw_clone)
    jobs_dict = get_job_dict(jobs_raw_clone)
    fit, res = sample.evaluator.eval(jobs_dict)
    w = hp.WEIGHT_ROBUSTNESS
    results['r'].append(res['r_mean'])
    results['s'].append(res['scom_mean'])
    results['sum'].append(res['scom_mean']+abs(res['r_mean']))
    results['sumw'].append(res['scom_mean']*(1-w)+abs(res['r_mean']*w))
    results['w'].append(w)
    results['obj'].append(obj)
    results['s_id'].append(s_id)


if __name__ == '__main__':
    
    objectives = [Objective.CMAX, Objective.F]
    #objectives = [Objective.CMAX]

    if ProcessSpawner.instances == {}:
        ProcessSpawner.instances["montecarlo"] = ProcessSpawner(func_target=sim.experiment_process, parallel_processes=hp.CPU_CORES)

    results = {'r': [], 's': [], 'sum': [], 'sumw': [], 'obj': [], 'w': [], 's_id': []}
    
    hp.N_MONTE_CARLO_EXPERIMENTS = 800
    hp.WEIGHT_ROBUSTNESS = 1
    i = 1
    for obj in objectives:
        hp.SCHED_OBJECTIVE = obj
        samples = base.load_samples(6, sample_ids=[0,1,2,3,4,6,8])
        samples = samples[:7]

        for sample in samples:
            s_id = samples.index(sample)
            jobs_raw_clone = deepcopy(sample.jobs_raw)
            
            # for ii in range(5):
            #     hp.WEIGHT_ROBUSTNESS = (ii+1)/5
            #     append_res(s_id, obj, jobs_raw_clone)

            #hp.WEIGHT_ROBUSTNESS = 1
            append_res(s_id, obj, jobs_raw_clone)

            print(f'{i}/{len(objectives)*len(samples)}')
            i += 1
    
    ProcessSpawner.instances["montecarlo"].kill_processes()
    df = pd.DataFrame(results)
    df = df[['r', 's', 'sum', 'sumw', 'w', 'obj', 's_id']] 
    #df.to_csv(p, sep=';', encoding='utf-8', header=True, index=False, decimal=",")
    descr = df.describe()
    print(df)
    print(descr)

    