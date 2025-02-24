from mpi4py import MPI
from utils.files import write_pkl
from utils.settings import TMP_DATA_DIR
from utils.logging import log_info, log_error
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def distribute_tasks(comm, task_queue, process_func, output=False, desc=None, node_workers=5):
    rank = comm.Get_rank()
    size = comm.Get_size()

    log_info(f"Total tasks: {task_queue.qsize()}")

    comm.Barrier()

    if rank == 0:
        master_process(comm, task_queue, size, node_workers)
    else:
        worker_process(comm, process_func, output, desc, node_workers)

    comm.Barrier()

def get_task(task_queue, num_active_workers, node_workers):
    total_tasks = task_queue.qsize()
    avg_tasks_per_worker = total_tasks / num_active_workers
    tasks_per_node = 0

    if avg_tasks_per_worker < 1:
        # If there are fewer tasks than workers, assign either 1 task or node_workers, whichever is smaller
        tasks_per_node = min(1, node_workers)
    else:
        # Distribute tasks evenly, but not exceeding node_workers or total tasks
        tasks_per_node = min(round(avg_tasks_per_worker), node_workers)

    # Ensure we don't try to get more tasks than are available in the queue
    tasks = [task_queue.get() for _ in range(min(tasks_per_node, task_queue.qsize()))]
    return tasks

def master_process(comm, task_queue, size, node_workers):
    try:
        active_workers = set(range(1, size))
        completed_tasks = 0
        total_tasks = task_queue.qsize()

        while active_workers:
            status = MPI.Status()
            worker_result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == 1:  # Worker is ready for new tasks
                if not task_queue.empty():
                    tasks = get_task(task_queue, len(active_workers), node_workers)
                    comm.send(tasks, dest=source, tag=1)
                    log_info(f"Sent {len(tasks)} tasks to worker {source}")
                else:
                    comm.send(None, dest=source, tag=2)  # No more tasks, terminate worker
                    active_workers.remove(source)
                    log_info(f"Worker {source} finished its jobs!")
            elif tag == 2:  # Worker completed tasks
                completed_tasks += worker_result
                log_info(f"Received results from worker {source}. Total completed: {completed_tasks}/{total_tasks}")

    except Exception as e:
        log_error(f"Master process encountered an error: {e}")

def worker_process(comm, process_func, output, desc, node_workers):
    try:
        rank = comm.Get_rank()
        local_results = []
    
        while True:
            comm.send(None, dest=0, tag=1)  # Request new tasks
            tasks = comm.recv(source=0, tag=MPI.ANY_TAG)
            if tasks is None:  # Termination signal
                break

            completed_tasks = 0
            log_info(f"Worker {rank} received {len(tasks)} tasks")  
            results = process_rank_tasks(tasks, process_func, node_workers)
            local_results.extend(results)
            completed_tasks += len(tasks)
            
            # Send completed tasks to rank 0
            comm.send(completed_tasks, dest=0, tag=2)

    except Exception as e:
        log_error(f"Worker {rank} encountered an error: {e}")

    if output:
        output_file = os.path.join(TMP_DATA_DIR, f'{desc}-rank{rank}.pkl')
        write_pkl(local_results, output_file)
    log_info(f"Worker {rank} completed all assigned tasks")


def process_rank_tasks(tasks, process_func, node_workers):
    try:
        results = []
        with ProcessPoolExecutor(max_workers=node_workers) as executor:
            futures = [executor.submit(process_func, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results
    except Exception as e:
        log_error(f"An error occurred in process_rank_tasks: {e}")
        raise e
