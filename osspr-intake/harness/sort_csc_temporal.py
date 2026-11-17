"""Baseline timing for ``torch_geometric.sampler.utils.sort_csc`` (temporal path).

Workload: a random graph with 200_000 nodes and 4_000_000 edges and an int64
node time per node. ``NeighborSampler(..., time_attr="time")`` calls
``to_csc`` -> ``sort_csc(row, col, src_node_time=...)``, which runs a
two-key stable ``lexsort``. The timed region is one ``sort_csc`` call.
End-to-end step: build a temporal ``NeighborLoader`` ([15, 10] neighbours,
batch 1024) and draw 20 batches; the script prints the sort_csc share of that step.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import torch

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler.utils import sort_csc

REPEATS = 9
N = 200_000
E = 4_000_000


def main() -> None:
    torch.manual_seed(0)
    row, col = torch.randint(0, N, (2, E))
    node_time = torch.randint(0, 10**6, (N,))

    def run():
        return sort_csc(row, col, src_node_time=node_time)

    r, c, perm = run()
    if not bool((c[1:] >= c[:-1]).all()) or not torch.equal(r, row[perm]) or not torch.equal(c, col[perm]):
        raise SystemExit("sort_csc output is not column-sorted")

    data = Data(x=torch.randn(N, 16), edge_index=torch.stack([row, col]), time=node_time)

    def step() -> None:
        loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=1024, time_attr="time",
                                input_nodes=torch.arange(20 * 1024), input_time=torch.full((20 * 1024,), 10**6))
        for _ in loader:
            pass

    step()
    samples_ms: list[float] = []
    step_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter(); run(); samples_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter(); step(); step_ms.append((time.perf_counter() - started) * 1000.0)
    median = statistics.median(samples_ms)
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    e2e = statistics.median(step_ms)
    print(f"baseline_median_ms={median:.4f}")
    print(f"baseline_noise_pct={noise:.4f}")
    print("runs_ms=" + ",".join(f"{s:.4f}" for s in samples_ms))
    print(f"e2e_loader_build_plus_20_batches_median_ms={e2e:.4f}")
    print(f"share_of_e2e_pct={median / e2e * 100.0:.2f}")


if __name__ == "__main__":
    main()
