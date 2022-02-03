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
    def preprocess_jobs(jobs_raw):
        """ use expected values for the operation durations. include uncertain processing times and machine failure probabilities. """
        i = 0
        for job in jobs_raw:
            for task in filter(lambda x: isinstance(x, list), job):
                if job[5] == "p1":
                    if task[0] == 1:
                        task[1] = int(
                            round(
                                Product1.M1DurationExp
                                + MachineFailure.M1FailureProb * MachineFailure.M1MaintenanceEffortExp
                            )
                        )
                    elif task[0] == 2:
                        task[1] = int(
                            round(
                                Product1.M2DurationExp
                                + MachineFailure.M2FailureProb * MachineFailure.M2MaintenanceEffortExp
                            )
                        )
                    elif task[0] == 3:
                        task[1] = int(
                            round(
                                Product1.M3DurationExp
                                + MachineFailure.M3FailureProb * MachineFailure.M3MaintenanceEffortExp
                            )
                        )
                elif job[5] == "p2":
                    if task[0] == 1:
                        task[1] = int(
                            round(
                                Product2.M1DurationExp
                                + MachineFailure.M1FailureProb * MachineFailure.M1MaintenanceEffortExp
                            )
                        )
                    elif task[0] == 2:
                        task[1] = int(
                            round(
                                Product2.M2DurationExp
                                + MachineFailure.M2FailureProb * MachineFailure.M2MaintenanceEffortExp
                            )
                        )
                    elif task[0] == 3:
                        task[1] = int(
                            round(
                                Product2.M3DurationExp
                                + MachineFailure.M3FailureProb * MachineFailure.M3MaintenanceEffortExp
                            )
                        )
            i += 1