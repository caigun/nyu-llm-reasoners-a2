import argparse
import timeit
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
# from a1_basics.model import BasicsTransformerLM
import a1_basics.model
import a1_basics.optimizer
from nsys_profile import annotated_scaled_dot_product_attention
import torch.cuda.nvtx as nvtx

def initialize_model(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int = 10000,
    context_length: int = 512,
    rope_theta: float = 10000.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """
    Initialize a transformer language model with the given hyperparameters.

    Args:
        d_model: Model embedding dimensionality
        d_ff: Feed-forward layer dimensionality
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size (default: 10000)
        context_length: Maximum sequence length (default: 512)
        rope_theta: RoPE theta value (default: 10000.0)
        device: Device to place model on (default: cuda if available, else cpu)

    Returns:
        Initialized transformer model
    """

    # Use the annotated version of attention to get NVTX profiling markers
    a1_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    model = a1_basics.model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.to(device)
    model.train()  # Set to training mode for backward pass

    return model


def generate_random_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random batch of input token IDs and target labels.

    Args:
        batch_size: Number of sequences in the batch
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (for random token generation)
        device: Device to place tensors on

    Returns:
        Tuple of (input_ids, target_labels)
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    # Shift labels: predict the next token
    target_labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    return input_ids, target_labels


def benchmark_forward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_steps: int = 100,
    n_warmup: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Benchmark the forward pass of the model.

    Args:
        model: The model to benchmark
        input_ids: Input token IDs of shape (batch_size, seq_length)
        n_steps: Number of steps to time (after warmup)
        n_warmup: Number of warmup steps before timing
        device: Device the model is on

    Returns:
        Dictionary containing timing statistics
    """
    model.eval()


    # Test timer on cpu first and then on hpc cuda
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)
            if device == "cuda" or device.startswith("cuda:"):
                torch.cuda.synchronize()

    # Timing
    def forward_step():
        with torch.no_grad():
            _ = model(input_ids)
        if device == "cuda" or device.startswith("cuda:"):
            torch.cuda.synchronize()

    timer = timeit.Timer(forward_step)
    times = timer.repeat(repeat=1, number=n_steps)
    total_time = times[0]

    return {
        "pass_type": "forward",
        "total_time_s": total_time,
        "avg_time_per_step_ms": (total_time / n_steps) * 1000,
        "steps_per_second": n_steps / total_time,
        "n_steps": n_steps,
        "n_warmup": n_warmup,
    }


def benchmark_forward_backward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_labels: torch.Tensor,
    n_steps: int = 100,
    n_warmup: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Benchmark both forward and backward passes of the model.

    Args:
        model: The model to benchmark
        input_ids: Input token IDs of shape (batch_size, seq_length)
        target_labels: Target labels for loss computation
        n_steps: Number of steps to time (after warmup)
        n_warmup: Number of warmup steps before timing
        device: Device the model is on

    Returns:
        Dictionary containing timing statistics
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Test timer on cpu first and then on hpc cuda
    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_labels.view(-1))
        loss.backward()
        optimizer.step()
        if device == "cuda" or device.startswith("cuda:"):
            torch.cuda.synchronize()

    # Timing
    def forward_backward_step():
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_labels.view(-1))
        loss.backward()
        optimizer.step()
        if device == "cuda" or device.startswith("cuda:"):
            torch.cuda.synchronize()


    timer = timeit.Timer(forward_backward_step)
    times = timer.repeat(repeat=1, number=n_steps)
    total_time = times[0]

    return {
        "pass_type": "forward+backward",
        "total_time_s": total_time,
        "avg_time_per_step_ms": (total_time / n_steps) * 1000,
        "steps_per_second": n_steps / total_time,
        "n_steps": n_steps,
        "n_warmup": n_warmup,
    }


def print_benchmark_results(results: Dict[str, Any]) -> None:
    """
    Pretty-print benchmark results.

    Args:
        results: Dictionary of benchmark results
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmark Results: {results['pass_type'].upper()}")
    print(f"{'=' * 70}")
    print(f"Total time:              {results['total_time_s']:.4f} seconds")
    print(f"Average time per step:   {results['avg_time_per_step_ms']:.2f} ms")
    print(f"Steps per second:        {results['steps_per_second']:.2f}")
    print(f"Number of steps:         {results['n_steps']}")
    print(f"Warmup steps:            {results['n_warmup']}")
    print(f"{'=' * 70}\n")


def profile_forward_pass_simple(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_iterations: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Simple forward pass for profiling with nsys.
    Use with: nsys profile --trace cuda,nvtx python benchmark.py --profile forward

    Args:
        model: The model to profile
        input_ids: Input token IDs
        n_iterations: Number of iterations to run
        device: Device to run on
    """
    model.eval()
    
    with torch.no_grad():
        for i in range(n_iterations):
            with nvtx.range(f"forward_iteration"):
                _ = model(input_ids)
                if device == "cuda" or device.startswith("cuda:"):
                    torch.cuda.synchronize()


def profile_forward_backward_pass_simple(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_labels: torch.Tensor,
    n_iterations: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Simple forward+backward pass for profiling with nsys.
    Use with: nsys profile --trace cuda,nvtx python benchmark.py --profile backward

    Args:
        model: The model to profile
        input_ids: Input token IDs
        target_labels: Target labels
        n_iterations: Number of iterations to run
        device: Device to run on
    """
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = a1_basics.optimizer.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    for i in range(n_iterations):
        with nvtx.range(f"backward_iteration"):
            optimizer.zero_grad()
            with nvtx.range(f"forward_pass_in_backward_iteration"):
                logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_labels.view(-1))
            with nvtx.range(f"backward_pass_in_backward_iteration"):
                loss.backward()
            with nvtx.range(f"optimizer_step"):
                optimizer.step()
            if device == "cuda" or device.startswith("cuda:"):
                torch.cuda.synchronize()



def main():
    parser = argparse.ArgumentParser(
        description="Benchmark transformer model forward and backward passes"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Model embedding dimensionality"
    )
    parser.add_argument(
        "--d_ff", type=int, default=2048, help="Feed-forward layer dimensionality"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=10000, help="Vocabulary size"
    )
    parser.add_argument(
        "--context_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for benchmarking"
    )
    parser.add_argument(
        "--seq_length", type=int, default=256, help="Sequence length for benchmarking"
    )
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of steps to time"
    )
    parser.add_argument(
        "--n_warmup", type=int, default=10, help="Number of warmup steps"
    )
    parser.add_argument(
        "--pass_type",
        type=str,
        choices=["forward", "backward", "both"],
        default="both",
        help="Type of pass to benchmark: forward only, backward only, or both",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["none", "forward", "backward", "both"],
        default="none",
        help="Profile with nsys: use 'nsys profile --trace cuda,nvtx python benchmark.py --profile {forward|backward|both}'",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("TRANSFORMER BENCHMARKING")
    print(f"{'=' * 70}")
    print(f"Device:                  {args.device}")
    print(f"Model Configuration:")
    print(f"  d_model:               {args.d_model}")
    print(f"  d_ff:                  {args.d_ff}")
    print(f"  num_layers:            {args.num_layers}")
    print(f"  num_heads:             {args.num_heads}")
    print(f"  vocab_size:            {args.vocab_size}")
    print(f"  context_length:        {args.context_length}")
    print(f"Data Configuration:")
    print(f"  batch_size:            {args.batch_size}")
    print(f"  seq_length:            {args.seq_length}")
    print(f"Benchmark Configuration:")
    print(f"  n_steps:               {args.n_steps}")
    print(f"  n_warmup:              {args.n_warmup}")
    print(f"  pass_type:             {args.pass_type}")
    print(f"  profiling:             {args.profile}")
    print(f"{'=' * 70}\n")

    # Initialize model
    print("Initializing model...")
    model = initialize_model(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        device=args.device,
    )
    print(f"Model initialized on device: {args.device}")

    # Generate random batch
    print("Generating random batch...")
    input_ids, target_labels = generate_random_batch(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        device=args.device,
    )
    print(f"Batch shape: {input_ids.shape}")

    # If profiling mode is enabled, run profiling instead of benchmarking
    if args.profile != "none":
        print("\n" + "=" * 70)
        print("PROFILING MODE - Run with: nsys profile --trace cuda,nvtx python benchmark.py --profile {mode}")
        print("=" * 70 + "\n")
        
        if args.profile in ["forward", "both"]:
            print("Profiling forward pass...")
            profile_forward_pass_simple(
                model=model,
                input_ids=input_ids,
                n_iterations=args.n_steps,
                device=args.device,
            )
            print("Forward pass profiling complete!")

        if args.profile in ["backward", "both"]:
            print("Profiling forward+backward pass...")
            profile_forward_backward_pass_simple(
                model=model,
                input_ids=input_ids,
                target_labels=target_labels,
                n_iterations=args.n_steps,
                device=args.device,
            )
            print("Forward+backward pass profiling complete!")
        
        return

    # Run benchmarks (non-profiling mode)
    if args.pass_type in ["forward", "both"]:
        print("\nRunning forward pass benchmark...")
        forward_results = benchmark_forward_pass(
            model=model,
            input_ids=input_ids,
            n_steps=args.n_steps,
            n_warmup=args.n_warmup,
            device=args.device,
        )
        print_benchmark_results(forward_results)

    if args.pass_type in ["backward", "both"]:
        print("Running forward+backward pass benchmark...")
        fb_results = benchmark_forward_backward_pass(
            model=model,
            input_ids=input_ids,
            target_labels=target_labels,
            n_steps=args.n_steps,
            n_warmup=args.n_warmup,
            device=args.device,
        )
        print_benchmark_results(fb_results)


if __name__ == "__main__":
    main()
