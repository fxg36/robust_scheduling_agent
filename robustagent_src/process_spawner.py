import multiprocessing
from multiprocessing.context import Process
import time


class ProcessSpawner:
    instances = {}

    def __init__(self, func_target, parallel_processes):
        man = multiprocessing.Manager()
        self.output_list = man.list()
        self.params = man.dict()
        self.params["run_process"] = man.list()
        self.no_processes = parallel_processes
        for _ in range(parallel_processes):
            self.params["run_process"].append(False)
        self.params['parallel_processes'] = parallel_processes
        self.processes = []

        global lock
        lock = multiprocessing.Lock()

        for p_id in range(parallel_processes):
            pr = Process(
                target=ProcessSpawner.target_caller,
                args=(
                    func_target,
                    p_id,
                    self.output_list,
                    self.params,
                    lock
                ),
                daemon=True,
            )
            self.processes.append(pr)
        for pr in self.processes:
            pr.start()

    @staticmethod
    def lock_init(l):
        global lock
        lock = l

    def clear_outputs(self):
        self.output_list[:] = []

    def kill_processes(self):
        for pr in self.processes:
            pr.kill()

    def activate_processes(self):
        for p in range(self.no_processes):
            self.params["run_process"][p] = True

    def await_processes(self, activate=True):
        if activate:
            self.activate_processes()
        while any(list(self.params["run_process"])):
            time.sleep(0.05)
        return list(self.output_list)

    @staticmethod
    def target_caller(func, proc_no, output_list, param, lock):
        while True:
            if not param["run_process"][proc_no]:
                time.sleep(0.005)
                continue
            result = func(proc_no, param, output_list)
            assert isinstance(result, list), 'result must be a list'
            with lock:
                output_list.extend(result)
                param["run_process"][proc_no] = False
