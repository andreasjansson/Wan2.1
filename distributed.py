import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from xfuser.core.distributed import get_sequence_parallel_world_size


class DistributedManager:
    """Manages distributed processing for the Wan inpainting model."""

    def __init__(self, model_class, model_args, start_method="spawn"):
        """Initialize the distributed manager.

        Args:
            model_class: The class of the model to instantiate
            model_args: Arguments to pass to the model constructor
        """
        self.model_class = model_class
        self.model_args = model_args

        self.world_size = torch.cuda.device_count()
        if self.world_size < 1:
            raise RuntimeError("No CUDA devices available")

        # Initialize multiprocessing components
        mp.set_start_method(start_method, force=True)
        self.model_ready_event = mp.Event()
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = []

    def setup(self):
        """Set up the distributed environment and start worker processes."""
        print(f"Setting up distributed model across {self.world_size} GPUs")

        # Start processes
        for rank in range(self.world_size):
            p = mp.Process(
                target=self.worker_process,
                args=(
                    rank,
                    self.world_size,
                    self.model_class,
                    self.model_args,
                    self.model_ready_event,
                    self.task_queue,
                    self.result_queue,
                ),
            )
            p.start()
            self.processes.append(p)

        # Wait for model to be initialized
        print("Waiting for model to be ready...")
        self.model_ready_event.wait()
        print("Model is ready")

    def submit_task(self, params):
        """Submit a task to be processed by worker processes.

        Args:
            params: Dictionary of parameters for the inpainting task

        Returns:
            The result from the worker process
        """
        # Generate a unique task ID
        task_id = str(time.time())

        # Submit task to queue for processing
        self.task_queue.put((task_id, params))

        # Wait for result
        result_id, result = self.result_queue.get()

        if isinstance(result, str) and result.startswith("Error:"):
            raise RuntimeError(result)

        return result

    def cleanup(self):
        """Clean up processes and distributed environment."""
        try:
            # Send exit signal to all processes
            for _ in range(self.world_size):
                self.task_queue.put(None)

            # Wait for processes to terminate
            for p in self.processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    @staticmethod
    def setup_dist(rank, world_size):
        """Initialize the distributed environment."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    @staticmethod
    def cleanup_dist():
        """Clean up distributed environment."""
        dist.destroy_process_group()

    @staticmethod
    def worker_process(
        rank, world_size, model_class, model_args, ready_event, task_queue, result_queue
    ):
        """Worker process that initializes the model and processes tasks.

        Args:
            rank: Process rank
            world_size: Total number of processes
            model_class: Class of the model to instantiate
            model_args: Arguments for model instantiation
            ready_event: Event to signal when model is ready
            task_queue: Queue for receiving tasks
            result_queue: Queue for sending results
        """
        try:
            DistributedManager.setup_dist(rank, world_size)
            torch.cuda.set_device(rank)

            # Create model args with the appropriate device and rank
            process_model_args = model_args.copy()
            process_model_args.update(
                {
                    "device_id": rank,
                    "rank": rank,
                    "dit_fsdp": True,
                    "t5_fsdp": True,
                    "use_usp": True,
                }
            )

            # Initialize model
            model = model_class(**process_model_args)

            # Signal that model is ready
            if rank == 0:
                print(
                    f"Model ready on rank {rank}, sequence parallel size: {get_sequence_parallel_world_size()}"
                )
                ready_event.set()

            # Process loop
            while True:
                # Wait for a task
                task = task_queue.get()
                if task is None:  # Exit signal
                    break

                # Unpack task parameters
                task_id, params = task

                try:
                    # Generate inpainted video
                    result = model.generate_inpaint(**params)

                    # Only rank 0 puts result in the result queue
                    if rank == 0:
                        result_queue.put((task_id, result))
                except Exception as e:
                    if rank == 0:
                        result_queue.put((task_id, f"Error: {str(e)}"))
                    print(f"Rank {rank} error: {str(e)}")

            DistributedManager.cleanup_dist()

        except Exception as e:
            print(f"Process {rank} failed: {str(e)}")
            if rank == 0:
                ready_event.set()  # Ensure event is set even if there's an error
                result_queue.put(
                    ("setup_error", f"Error setting up model on rank {rank}: {str(e)}")
                )
