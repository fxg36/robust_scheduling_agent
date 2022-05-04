import random

class Product1:
    M1Duration = (149, 150, 155)  # triangular distributed uncertain processing time
    M2Duration = (165, 167, 170)
    M3Duration = (90, 95, 102)
    M1DurationExp = (M1Duration[0] + M1Duration[1] + M1Duration[2]) / 3
    M2DurationExp = (M2Duration[0] + M2Duration[1] + M2Duration[2]) / 3
    M3DurationExp = (M3Duration[0] + M3Duration[1] + M3Duration[2]) / 3
    ProcessingTimeExpected = M1DurationExp + M2DurationExp + M3DurationExp


class Product2:
    M1Duration = (165, 168, 172)
    M2Duration = (205, 207, 212)
    M3Duration = (195, 198, 200)
    M1DurationExp = (M1Duration[0] + M1Duration[1] + M1Duration[2]) / 3
    M2DurationExp = (M2Duration[0] + M2Duration[1] + M2Duration[2]) / 3
    M3DurationExp = (M3Duration[0] + M3Duration[1] + M3Duration[2]) / 3
    ProcessingTimeExpected = M1DurationExp + M2DurationExp + M3DurationExp


class MachineFailure:
    M1FailureProb = 0.05
    M2FailureProb = 0.15
    M3FailureProb = 0.1
    M1MaintenanceEffort = (35, 40, 48)
    M2MaintenanceEffort = (25, 27, 30)
    M3MaintenanceEffort = (30, 35, 40)
    M1MaintenanceEffortExp = (M1MaintenanceEffort[0] + M1MaintenanceEffort[1] + M1MaintenanceEffort[2]) / 3
    M2MaintenanceEffortExp = (M2MaintenanceEffort[0] + M2MaintenanceEffort[1] + M2MaintenanceEffort[2]) / 3
    M3MaintenanceEffortExp = (M3MaintenanceEffort[0] + M3MaintenanceEffort[1] + M3MaintenanceEffort[2]) / 3


class UnplannedDemand:  # currently not in use
    UnplannedJobArrivalProb = 0.4
    PriorityIncreasementProb = 0.2


class JobFactory:
    """1) Generating random flow shop jobs
    2) Preprocessing jobs to include uncertainties for simulation"""

    @staticmethod
    def get_random_jobs(no_jobs_p1: int, no_jobs_p2: int):
        product_jobs = [1] * no_jobs_p1 + [2] * no_jobs_p2
        random.shuffle(product_jobs)
        job_array = []
        buffer = (Product1.M2DurationExp + Product2.M2DurationExp) / 2
        i = 0
        for j in product_jobs:
            if j == 1:
                job_array.append(
                    [
                        [1, int(round(Product1.M1DurationExp))],
                        [2, int(round(Product1.M2DurationExp))],
                        [3, int(round(Product1.M3DurationExp))],
                        Product1.ProcessingTimeExpected + i * buffer,
                        random.randint(1, 2),
                        "p1",
                    ]
                )
            elif j == 2:
                job_array.append(
                    [
                        [1, int(round(Product2.M1DurationExp))],
                        [2, int(round(Product2.M2DurationExp))],
                        [3, int(round(Product2.M3DurationExp))],
                        Product2.ProcessingTimeExpected + i * buffer,
                        random.randint(1, 2),
                        "p2",
                    ]
                )
            i += 1

        return job_array

    @staticmethod
    def preprocess_one_operation(jobs_raw, job_id, machine_id, do_round=True):
        def get_dist_values(product, machine):
            if machine == 1:
                b = (
                    MachineFailure.M1MaintenanceEffort[0],
                    MachineFailure.M1MaintenanceEffort[1],
                    MachineFailure.M1MaintenanceEffort[2],
                )
                p = MachineFailure.M1FailureProb
                if product == "p1":
                    a = (Product1.M1Duration[0], Product1.M1Duration[1], Product1.M1Duration[2])
                else:
                    a = (Product2.M1Duration[0], Product2.M1Duration[1], Product2.M1Duration[2])
            elif machine == 2:
                b = (
                    MachineFailure.M2MaintenanceEffort[0],
                    MachineFailure.M2MaintenanceEffort[1],
                    MachineFailure.M2MaintenanceEffort[2],
                )
                p = MachineFailure.M2FailureProb
                if product == "p1":
                    a = (Product1.M2Duration[0], Product1.M2Duration[1], Product1.M2Duration[2])
                else:
                    a = (Product2.M2Duration[0], Product2.M2Duration[1], Product2.M2Duration[2])
            else:  # 3
                b = (
                    MachineFailure.M3MaintenanceEffort[0],
                    MachineFailure.M3MaintenanceEffort[1],
                    MachineFailure.M3MaintenanceEffort[2],
                )
                p = MachineFailure.M3FailureProb
                if product == "p1":
                    a = (Product1.M3Duration[0], Product1.M3Duration[1], Product1.M3Duration[2])
                else:
                    a = (Product2.M3Duration[0], Product2.M3Duration[1], Product2.M3Duration[2])

            def e(tuple):
                return (tuple[0] + tuple[1] + tuple[2]) / 3

            def std(tuple):
                return ((tuple[0] - tuple[2]) ** 2 + (tuple[2] - tuple[1]) ** 2 + (tuple[0] - tuple[1]) ** 2) ** 0.5 / 6

            return e(a) - std(a), \
                e(a), \
                e(a) + p * e(b), \
                e(a) + p * e(b) + std(a)/4 + p * std(b)/4, \
                e(a) + p * e(b) + std(a)/1 + p * std(b)/1

        job = jobs_raw[job_id - 1]
        product = product = job[5]
        values = get_dist_values(product, machine_id)
        values = sorted(list(values))

        if do_round:
            values = list(map(lambda x: int(round(x)), values))

        return values

    @staticmethod
    def preprocess_jobs(jobs_raw, do_round=True, expected_value_idx = 2):
        """ use expected values for the operation durations. include uncertain processing times and machine failure probabilities. """

        for job in jobs_raw:
            job_id = jobs_raw.index(job) + 1
            for task in filter(lambda x: isinstance(x, list), job):
                val = JobFactory.preprocess_one_operation(jobs_raw, job_id=job_id, machine_id=task[0], do_round=do_round)
                task[1] = val[expected_value_idx]
