import math
import subprocess
def determine_gpus():
    try:
        out = subprocess.check_output(['nvidia-smi', '--list-gpus'])
        return len(out.decode().splitlines())
    except Exception:
        return 0

def divisors(n):
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n//i)
    return sorted(divs)

def recover_function(tensor_size, pipeline_size, hidden_size, num_layers, attention_heads):
    gpu_count = determine_gpus()
    if gpu_count == 0:
        raise RuntimeError("No GPUs detected")

    desired = (tensor_size, pipeline_size)
    candidates = []

    # Tensor splits must divide hidden_size and attention_heads
    for tp in divisors(hidden_size):
        if attention_heads % tp != 0:
            continue

        # Pipeline splits must divide num_layers
        for pp in divisors(num_layers):
            world_size = tp * pp
            if world_size > gpu_count:
                continue
            if gpu_count % world_size != 0:
                continue

            dp = gpu_count // world_size
            candidates.append((tp, pp, dp))

    if not candidates:
        # as a last resort, default to pure data parallel
        return [(1, 1, gpu_count)]

    def score(cfg):
        tp, pp, _ = cfg
        return abs(tp - desired[0]) + abs(pp - desired[1])

    best = min(candidates, key=score)
    return [desired, (best[0], best[1], best[2])]
