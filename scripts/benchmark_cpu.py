"""Measure representative fine-tuning throughput before committing compute."""
import time

import torch

import competition as c

c.seed_all()
model = c.initialize_tail()
model.train()
x = torch.randn(8, 128, 384)
mask = torch.ones(8, 128, dtype=torch.bool)
rows = []
for threads in (2, 4, 8):
    torch.set_num_threads(threads)
    started = time.perf_counter()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        model(x, mask).square().mean().backward()
    elapsed = (time.perf_counter()-started)/3
    rows.append({"threads":threads,"seconds_per_8x128_training_microbatch":elapsed})
    print(rows[-1], flush=True)
c.dump(c.ROOT / "artifacts/cpu_benchmark.json", rows)
