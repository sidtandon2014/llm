import torch
import os
from transformers import TrainerCallback
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist

class ProfilerCallback(TrainerCallback):
    """
    A custom callback to integrate the PyTorch profiler with the Hugging Face Trainer.
    """
    def __init__(self, output_dir):
        self.profiler = None
        self.output_dir = output_dir
        self.schedule = torch.profiler.schedule(skip_first=1, wait=0, warmup=1, active=2, repeat=1)
        # Create a profiler context manager that will be active during training
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], # Profile both CPU and GPU
            schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=1, active=2, repeat=1), # A schedule for profiling steps
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profile'), # Where to save the trace
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        )

    def on_train_begin(self, args, state, control, **kwargs):
        """Starts the profiler when training begins."""
        print("Starting profiler...")
        
        # Only profile on the main process (rank 0)
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
            
        if rank == 0:
            print("Starting profiler on rank 0...")
            # Ensure the output directory for traces exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=self.schedule,
                # Use a handler that saves the trace to a unique file for this rank
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(self.output_dir, f"profile/profile_rank_{rank}")
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )
            self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        """Advances the profiler to the next step after each training step."""
        if self.profiler:
            print("Profiling step")
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        """Stops the profiler when training ends."""
        if self.profiler:
            print("Stopping profiler...")
            self.profiler.__exit__(None, None, None)