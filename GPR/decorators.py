import multiprocessing as mp
from multiprocessing import Manager
import numpy as np

def multiprocess(*args, **kwargs):
    """Decorator method for multiprocessing an embarassingly parallel function.
    Please refer to https://docs.python.org/3/library/multiprocessing.html#multiprocessing.sharedctypes.multiprocessing.Manager
    for manager types.

    """

    def wrapper(function):
        n = len(kwargs['iterable'])
        print("Number of CPUs: %s"%(mp.cpu_count()))
        p = mp.Pool(processes=mp.cpu_count())
        print(f"Number of processes: {n}")
        jobs = []
        for iter in kwargs['iterable']:
            process = p.Process(target=function, args=(iter,))
            jobs.append(process)
            jobs[-1].start()
            active_processors = [jobs[i].is_alive() for i in range(len(jobs))]
            if (len(active_processors) == mp.cpu_count()-1) and all(active_processors) == True:
                while all(active_processors) == True:
                    active_processors = [jobs[i].is_alive() for i in range(len(jobs))]
                inactive = int(np.where(np.array(active_processors) == False)[0])
                jobs[inactive].terminate()
                jobs.remove(jobs[inactive])

        for job in jobs:
            job.join()
        p.close()
    return wrapper




