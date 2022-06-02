from random import randint
import hyperparam as hp
from robustness_evaluator import BaselineSchedule
import pickle
from pathlib import Path
from data import JobFactory
from typing import List

def generate_samples(n_jobs,use_train_samples):
    samples = []
    for _ in range(hp.N_SAMPLES):
        no_jobs_p1 = randint(1, n_jobs - 1)
        random_jobvector = JobFactory.get_random_jobs(no_jobs_p1, n_jobs - no_jobs_p1)
        bs = BaselineSchedule(random_jobvector)
        samples.append(bs)
        print("CREATED 1 SCHEDULE")

    p = Path(".")
    if use_train_samples:
        p = p / "samples" / "train" / f"samples{hp.N_SAMPLES}_jobs{n_jobs}_{hp.SCHED_OBJECTIVE}.pickle"
    else:
        p = p / "samples" / "test" / f"samples{hp.N_SAMPLES}_jobs{n_jobs}_{hp.SCHED_OBJECTIVE}.pickle"
        
    with open(p, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples(n: int, use_train_samples=False) -> List[BaselineSchedule]:
    if n == 0:
        samples = []
        samples.extend(load_samples(5, use_train_samples))
        samples.extend(load_samples(10, use_train_samples))
        return samples
    else:
        p = Path(".")
        if use_train_samples:
            p = p / "samples" / "train" / f"samples{hp.N_SAMPLES}_jobs{n}_{hp.SCHED_OBJECTIVE}.pickle"
        else:
            p = p / "samples" / "test" / f"samples{hp.N_SAMPLES}_jobs{n}_{hp.SCHED_OBJECTIVE}.pickle"
        try:
            with open(p, "rb") as f:
                samples = pickle.load(f)
            return samples
        except:
            generate_samples(n,use_train_samples)
            return load_samples(n)