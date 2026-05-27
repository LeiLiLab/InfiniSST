"""Benchmark FAISS retrieval latency for different database sizes.

Measures ONLY index.search() time, excludes query generation and normalization.
Tests: IndexFlatIP on CPU vs GPU, database sizes 100/1k/10k,
       single-vector query vs MaxSim (24 windows sequential vs batched).
"""
import time
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu or faiss-gpu required")

# ======Configuration=====
VECTOR_DIM = 1024
DB_SIZES = [100, 1_000, 10_000]
TOP_K = 10
N_WARMUP = 50
N_TRIALS = 1000
N_MAXSIM_WINDOWS = 24
# ======Configuration=====


def build_index_cpu(db_size: int, dim: int) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(dim)
    vectors = np.random.randn(db_size, dim).astype(np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def build_index_gpu(db_size: int, dim: int):
    cpu_index = build_index_cpu(db_size, dim)
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return gpu_index, res
    except Exception as e:
        print(f"  [WARN] GPU index failed: {e}")
        return None, None


def bench_search_single(index, queries: np.ndarray, top_k: int, n_warmup: int, n_trials: int) -> float:
    """Single-vector search. queries: [N_TRIALS, 1, D]. Returns mean µs."""
    for i in range(n_warmup):
        index.search(queries[i % len(queries)], top_k)

    start = time.perf_counter()
    for i in range(n_trials):
        index.search(queries[i % len(queries)], top_k)
    elapsed = time.perf_counter() - start
    return (elapsed / n_trials) * 1e6


def bench_search_maxsim_seq(index, queries_batch: np.ndarray, top_k: int, n_warmup: int, n_trials: int) -> float:
    """MaxSim sequential: W individual queries. queries_batch: [N, W, D]. Returns mean µs per chunk."""
    for w in range(n_warmup):
        q = queries_batch[w % len(queries_batch)]
        for j in range(q.shape[0]):
            index.search(q[j:j+1], top_k)

    start = time.perf_counter()
    for i in range(n_trials):
        q = queries_batch[i % len(queries_batch)]
        for j in range(q.shape[0]):
            index.search(q[j:j+1], top_k)
    elapsed = time.perf_counter() - start
    return (elapsed / n_trials) * 1e6


def bench_search_maxsim_batched(index, queries_batch: np.ndarray, top_k: int, n_warmup: int, n_trials: int) -> float:
    """MaxSim batched: all W windows in one search call. queries_batch: [N, W, D]. Returns mean µs per chunk."""
    for w in range(n_warmup):
        index.search(queries_batch[w % len(queries_batch)], top_k)

    start = time.perf_counter()
    for i in range(n_trials):
        index.search(queries_batch[i % len(queries_batch)], top_k)
    elapsed = time.perf_counter() - start
    return (elapsed / n_trials) * 1e6


def main():
    print(f"FAISS Retrieval Latency Benchmark (search-only, no encoding)")
    print(f"  Vector dim:      {VECTOR_DIM}")
    print(f"  Top-K:           {TOP_K}")
    print(f"  MaxSim windows:  {N_MAXSIM_WINDOWS}")
    print(f"  Warmup:          {N_WARMUP}")
    print(f"  Trials:          {N_TRIALS}")

    has_gpu = hasattr(faiss, 'StandardGpuResources')
    print(f"  GPU support:     {has_gpu}")
    print(flush=True)
    import sys

    # Pre-generate all queries (exclude from timing)
    n_gen = max(N_TRIALS, 5000)
    single_queries = np.random.randn(n_gen, 1, VECTOR_DIM).astype(np.float32)
    for i in range(n_gen):
        faiss.normalize_L2(single_queries[i])

    maxsim_queries = np.random.randn(n_gen, N_MAXSIM_WINDOWS, VECTOR_DIM).astype(np.float32)
    for i in range(n_gen):
        faiss.normalize_L2(maxsim_queries[i])

    print(f"{'DB Size':>8} | {'Mode':<24} | {'Dev':>4} | {'Mean µs':>10} | {'Mean ms':>10} | Note")
    print("-" * 95)
    sys.stdout.flush()

    for db_size in DB_SIZES:
        cpu_index = build_index_cpu(db_size, VECTOR_DIM)

        lat = bench_search_single(cpu_index, single_queries, TOP_K, N_WARMUP, N_TRIALS)
        print(f"{db_size:>8} | {'Single vector':<24} | {'CPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | 1 search")

        lat = bench_search_maxsim_seq(cpu_index, maxsim_queries, TOP_K, N_WARMUP, N_TRIALS // 5)
        print(f"{db_size:>8} | {'MaxSim seq (W=24)':<24} | {'CPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | 24x search loop")

        lat = bench_search_maxsim_batched(cpu_index, maxsim_queries, TOP_K, N_WARMUP, N_TRIALS // 5)
        print(f"{db_size:>8} | {'MaxSim batch (W=24)':<24} | {'CPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | 1x search(24,D)")

        if has_gpu:
            gpu_index, gpu_res = build_index_gpu(db_size, VECTOR_DIM)
            if gpu_index is not None:
                lat = bench_search_single(gpu_index, single_queries, TOP_K, N_WARMUP, N_TRIALS)
                print(f"{db_size:>8} | {'Single vector':<24} | {'GPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | 1 search")

                lat = bench_search_maxsim_seq(gpu_index, maxsim_queries, TOP_K, N_WARMUP, N_TRIALS // 5)
                print(f"{db_size:>8} | {'MaxSim seq (W=24)':<24} | {'GPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | 24x search loop")

                lat = bench_search_maxsim_batched(gpu_index, maxsim_queries, TOP_K, N_WARMUP, N_TRIALS // 5)
                print(f"{db_size:>8} | {'MaxSim batch (W=24)':<24} | {'GPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | 1x search(24,D)")

                del gpu_index, gpu_res

        print()

    # Also benchmark raw numpy/torch matmul for comparison
    print("--- Raw matmul comparison (no FAISS overhead) ---")
    import torch
    for db_size in DB_SIZES:
        db = torch.randn(db_size, VECTOR_DIM, dtype=torch.float32)
        db = db / db.norm(dim=1, keepdim=True)

        # Single vector CPU torch
        q = torch.randn(1, VECTOR_DIM, dtype=torch.float32)
        q = q / q.norm(dim=1, keepdim=True)
        for _ in range(N_WARMUP):
            _ = (q @ db.T).topk(TOP_K)
        start = time.perf_counter()
        for _ in range(N_TRIALS):
            _ = (q @ db.T).topk(TOP_K)
        elapsed = time.perf_counter() - start
        lat = (elapsed / N_TRIALS) * 1e6
        print(f"{db_size:>8} | {'torch matmul+topk':<24} | {'CPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | [1,D]x[D,N]")

        # MaxSim batched CPU torch
        q = torch.randn(N_MAXSIM_WINDOWS, VECTOR_DIM, dtype=torch.float32)
        q = q / q.norm(dim=1, keepdim=True)
        for _ in range(N_WARMUP):
            sims = q @ db.T
            _ = sims.max(dim=0).values.topk(TOP_K)
        start = time.perf_counter()
        for _ in range(N_TRIALS):
            sims = q @ db.T
            _ = sims.max(dim=0).values.topk(TOP_K)
        elapsed = time.perf_counter() - start
        lat = (elapsed / N_TRIALS) * 1e6
        print(f"{db_size:>8} | {'torch MaxSim matmul':<24} | {'CPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | [W,D]x[D,N]+max+topk")

        # GPU torch
        if torch.cuda.is_available():
            db_g = db.cuda()
            q_g = q.cuda()
            torch.cuda.synchronize()
            for _ in range(N_WARMUP):
                sims = q_g @ db_g.T
                _ = sims.max(dim=0).values.topk(TOP_K)
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(N_TRIALS):
                sims = q_g @ db_g.T
                _ = sims.max(dim=0).values.topk(TOP_K)
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            lat = (elapsed / N_TRIALS) * 1e6
            print(f"{db_size:>8} | {'torch MaxSim matmul':<24} | {'GPU':>4} | {lat:>10.1f} | {lat/1000:>10.4f} | [W,D]x[D,N]+max+topk")

        print()

    print("=" * 95)
    print(f"Vector dim = {VECTOR_DIM}. MaxSim = {N_MAXSIM_WINDOWS} windows per 1.92s chunk.")
    print("Audio encoding ≈ 100-200ms. Retrieval should be << encoding time.")


if __name__ == "__main__":
    main()
