#!/usr/bin/env python3
"""Prompt banks for the RouteScout K4 scaling corpus (EXP-078).

Three mutually disjoint banks:

- ``TRAIN_BANK``  — pool prompts. Runs are drawn deterministically in bank
  order, so every scale point (5k/10k/25k/50k generated tokens) is a **nested
  prefix** of one corpus. The bank is assembled by round-robin across domains,
  so even the smallest prefix already spans every topic: without that, a 5k-token
  point would be a Rust/C++ corpus and the scaling curve would be measuring
  topic, not data volume.
- ``VAL_BANK``    — model-selection prompts. Never trained on, at any scale.
- ``TEST_BANK``   — final unseen evaluation. Never used for training or model
  selection; touched only by the Phase-3 comparison.

Domain groups: Rust, C/C++, compilers/SSA/codegen, concurrency, lock-free data
structures, OS + memory management, SSD/NVMe/mmap I/O, networking, distributed
systems, databases, mathematical reasoning, algorithms, ML/MoE architecture,
inference optimization, ordinary explanatory prose, structured reasoning, code
generation/debugging, and systems-perf measurement.
"""

from __future__ import annotations

# --- pool, grouped by domain (ordering is applied below) ----------------

_DOMAINS: dict[str, list[str]] = {
    "rust": [
        "Explain Rust's borrow checker to an experienced C++ programmer, then show a case where NLL accepts code that a lexical checker would reject.",
        "Write a Rust iterator adapter that lazily scans a byte stream for a delimiter and returns borrowed slices, and explain the lifetime bound you needed.",
        "Compare Arc<Mutex<T>>, RwLock, and a sharded lock table for a hot in-memory index. Show the contention analysis, not just the conclusion.",
        "Implement a Rust trait object dispatch table by hand and explain where dyn dispatch actually costs you versus a generic monomorphization.",
        "Explain Pin and self-referential structs in Rust, then show why an async state machine needs both.",
        "Write a Rust build script and explain when proc-macro expansion becomes the dominant compile-time cost.",
        "Explain how Rust's Drop order works for a struct containing a Vec and a guard, and what a panic during Drop changes.",
        "Describe how to write a Rust FFI boundary that passes a borrowed slice to C without copying and without permitting an aliasing bug.",
        "Explain the difference between a Rust error type implemented with thiserror and one built from an enum, in terms of API stability.",
        "Show how to implement an intrusive linked list in Rust safely with a raw pointer and explain the invariant you must maintain.",
        "Explain how Rust's slice patterns and matches! macro let you write a state machine without a large match on enums.",
        "Describe a Rust benchmark harness that avoids the optimizer eliminating the measured work.",
    ],
    "cpp": [
        "Write a C function that parses a length-prefixed binary framing protocol and never reads past the buffer, and justify each bounds check.",
        "Explain strict aliasing and effective type in C, and show three real optimizations the compiler can only perform if you respect them.",
        "Implement a C++ arena allocator with alignment guarantees for SIMD types, and discuss destructor handling for non-trivial objects.",
        "Explain the C++ object model for multiple inheritance, including vtable layout and pointer adjustment on a secondary base.",
        "Show how to write a C++20 concepts-constrained template that gives better diagnostics than an SFINAE overload set.",
        "Discuss signed integer overflow as undefined behaviour in C and how a compiler may legally use it to prove a loop terminates.",
        "Explain C++ move semantics for a type that owns a raw buffer, including what the moved-from state must guarantee.",
        "Describe how to implement a small-vector optimization in C++ without violating strict aliasing.",
        "Explain the rules for template argument deduction for forwarding references and how std::forward preserves value category.",
        "Discuss how C++ exception unwinding interacts with destructors in a lock-holding scope and what RAII must guarantee.",
        "Explain how to detect and avoid a dangling reference returned from a C++ initializer-list constructor.",
        "Describe a C++ approach to representing a tagged union that is safer than reinterpret_cast on a byte buffer.",
    ],
    "compilers": [
        "Walk through SSA construction for a program with a loop and an if-then-else, including where phi functions appear and why.",
        "Explain how a register allocator models interference and where graph colouring breaks down for real machine constraints.",
        "Describe instruction selection by tree tiling, and explain how a dynamic-programming tiler handles multiple addressing modes.",
        "Explain loop-invariant code motion and the conditions under which it is unsafe because of exceptions or volatile accesses.",
        "How does an optimizing compiler decide to inline a function, and how do profile-guided hints change that decision?",
        "Explain the difference between a parser combinator library and a table-driven LR parser in error recovery quality.",
        "Explain how a compiler implements a closure by lifting captured variables, and where escape analysis changes the allocation.",
        "Describe how an ahead-of-time compiler handles separately compiled modules with inlining across boundaries.",
        "Explain how loop unrolling interacts with the instruction cache and when unrolling hurts performance.",
        "Describe how a compiler lowers exceptions into unwind tables and why that affects optimization.",
        "Explain how constant folding and algebraic simplification can change floating-point results and why that may be disallowed.",
        "Describe how a linker performs section garbage collection and why it requires function-sections.",
    ],
    "concurrency": [
        "Design a lock-free bounded MPMC queue and explain the memory ordering you need on ARM64 for the sequence-number scheme.",
        "Explain why a compare-and-swap loop can suffer from the ABA problem and two different ways to make it safe.",
        "Describe hazard pointers and epoch-based reclamation, and compare their latency profiles under a read-heavy workload.",
        "Write a seqlock for a small read-mostly configuration struct and prove that readers cannot observe a torn value.",
        "Explain false sharing in a cache-coherent multicore and show how padding a counter array changes throughput.",
        "Discuss why memory ordering is cheaper on x86 than on ARM, and what a release store actually compiles to on AArch64.",
        "Explain the difference between a mutex and a spinlock for a critical section that can block on I/O.",
        "Describe how a thread pool should size itself for a CPU-bound workload and why oversubscription hurts.",
        "Explain how a condition variable avoids lost wakeups and why the predicate must be checked in a loop.",
        "Discuss how a reader-writer lock can starve writers and what a fair implementation must do.",
        "Explain how a work-stealing scheduler balances load and what makes a task too small to steal profitably.",
        "Describe how to detect a deadlock in a lock hierarchy and how to order acquisitions to avoid it.",
    ],
    "os_memory": [
        "Explain demand paging, page faults, and why a minor fault is much cheaper than a major fault on a modern kernel.",
        "Compare a bump allocator, a segregated free list, and a slab allocator for an object pool with mixed sizes.",
        "Explain copy-on-write fork, its interaction with a large resident process, and how to avoid the worst case.",
        "Describe how a kernel handles a page fault on a file-backed mapping and when it chooses readahead versus a synchronous read.",
        "Explain huge pages, transparent huge page compaction, and why fragmentation hurts latency more than throughput.",
        "Discuss NUMA locality for a multi-socket machine and how a first-touch policy shapes placement.",
        "Explain how a memory-mapped ring buffer between a process and a device is made coherent and when barriers are needed.",
        "Describe how an operating system reclaims page cache under pressure and why it may evict the wrong pages.",
        "Explain how a signal handler can safely interrupt a blocked system call and what EINTR requires the caller to do.",
        "Describe how swapping interacts with memory-mapped model weights that are already backed by a file.",
        "Explain how a kernel implements a futex wait and why it avoids a syscall in the uncontended case.",
        "Discuss how memory overcommit policy changes the failure mode of a large allocation.",
    ],
    "storage_io": [
        "Compare mmap, pread, and Linux async I/O for streaming a model larger than RAM from an NVMe SSD, and quantify where each one stalls.",
        "Explain NVMe queue depth, submission and completion queues, and why queue depth matters less than latency for a single-threaded reader.",
        "Discuss direct I/O versus buffered I/O for a workload that must not evict a useful page cache working set.",
        "Explain the difference between SSD wear levelling, write amplification, and garbage collection stalls, and how each shows up in tail latency.",
        "Design a two-tier hot/warm cache for expert weights on a 16 GB machine backed by a slower SSD, and justify the eviction policy.",
        "Explain why random 4 KiB reads from an SSD can be ten times slower than sequential reads even when the device claims high IOPS.",
        "Describe how a filesystem's block allocator affects fragmentation and why a heavily fragmented file reads slower.",
        "Explain how to measure a storage device's latency distribution without the page cache hiding the real device behaviour.",
        "Compare a log-structured file layout with an in-place updated layout for a write-heavy workload on flash.",
        "Describe how to overlap computation with storage reads in a single-threaded decoder without introducing data races.",
        "Explain why readahead heuristics can hurt a workload with a poor spatial locality pattern.",
        "Describe how a storage engine batches many small reads into fewer larger ones and the tradeoff it introduces.",
    ],
    "networking": [
        "Explain TCP congestion control from slow start through CUBIC, and where an application-level backpressure scheme beats it.",
        "Compare epoll, io_uring, and a thread-per-connection design for ten thousand mostly idle sockets.",
        "Explain head-of-line blocking in HTTP/1.1 and how HTTP/2 multiplexing solves it while creating a new problem.",
        "Discuss Nagle's algorithm, delayed ACKs, and TCP_NODELAY in the context of a request-response protocol with small messages.",
        "Explain QUIC's use of UDP, connection migration, and how it avoids the handshake cost of TCP plus TLS.",
        "Describe a protocol for streaming large model outputs over a lossy link, including backpressure and resumability.",
        "Explain how a DNS resolver caches results and why TTL handling affects failover behaviour.",
        "Describe how a proxy should handle a slow upstream and a fast downstream without buffering the whole response.",
        "Explain how TLS session resumption reduces handshake cost and what it must not weaken.",
        "Describe how to design an RPC protocol so a client can safely retry a request whose response was lost.",
        "Explain how a load balancer's health check can cause a flapping backend and how to damp it.",
        "Discuss how packet-level pacing differs from application-level rate limiting for smooth throughput.",
    ],
    "distributed": [
        "Explain the Raft election and log-replication rules, and describe a scenario where a naive implementation loses a committed entry.",
        "Compare a leader-based replication scheme with a quorum scheme for tail latency under cross-region writes.",
        "Explain how a distributed system detects failure without a shared clock, and why a perfect failure detector is impossible.",
        "Discuss exactly-once semantics for a stream processor and explain what it actually means for a non-transactional sink.",
        "Explain consistent hashing, virtual nodes, and why rebalancing cost matters more than key distribution quality.",
        "Describe a two-phase commit coordinator and three concrete ways it can block indefinitely.",
        "Explain how a distributed key-value store implements a linearizable read and what it costs versus a stale read.",
        "Describe how a gossip protocol converges and how long convergence takes after a large membership change.",
        "Explain how a distributed scheduler avoids a thundering herd when a shared resource recovers.",
        "Discuss how to implement idempotent request handling across retries in a service with at-least-once delivery.",
        "Explain the difference between a logical clock and a vector clock and what each can and cannot order.",
        "Describe how a distributed cache invalidates an entry when the source of truth changes in another region.",
    ],
    "databases": [
        "Explain the difference between a B-tree and an LSM-tree for a write-heavy workload, including read amplification and compaction cost.",
        "Walk through how a database implements MVCC and how long-running readers cause version retention problems.",
        "Explain write-ahead logging, group commit, and fsync placement for a crash-safe storage engine.",
        "Discuss query planning for a join across three tables, and explain when a hash join beats a sort-merge join.",
        "Explain how a columnar store compresses integers with run-length and dictionary encoding, and what that does to scan speed.",
        "Describe how a database's buffer pool replacement policy interacts with an OS page cache holding the same pages.",
        "Explain how an index-only scan works and what makes it inapplicable for a particular query.",
        "Describe how a database handles a transaction that must read its own uncommitted writes under MVCC.",
        "Explain how a query optimizer uses statistics histograms and where a stale histogram causes a bad plan.",
        "Discuss how a database implements a skip-scan index for a composite key with a missing leading column.",
        "Explain how a storage engine implements a range delete on an LSM-tree without rewriting every overlapping file.",
        "Describe how a database guarantees durability when the filesystem lies about a completed fsync.",
    ],
    "math": [
        "Derive the gradient of softmax cross-entropy and explain why subtracting the maximum logit is numerically stable.",
        "Prove that a finite-state Markov chain with a positive recurrent class has a unique stationary distribution, and give the intuition.",
        "Work through a biased-coin Bayesian update numerically, starting from a uniform prior and three observed heads.",
        "Explain why the determinant of a matrix equals the product of its eigenvalues, using the characteristic polynomial.",
        "Show that the sum of the first n cubes equals the square of the sum of the first n integers, and give a combinatorial proof.",
        "Explain Lagrange multipliers and when the method fails because a constraint qualification does not hold.",
        "Compute the expected number of trials until two consecutive successes in an independent Bernoulli sequence, step by step.",
        "Explain the difference between a probability density and a probability mass, and where the Jacobian appears in a change of variables.",
        "Prove that a symmetric matrix has an orthonormal eigenbasis and explain where the proof uses symmetry.",
        "Explain the central limit theorem's conditions and give a counterexample when the variance is infinite.",
        "Derive the bias-variance decomposition for squared error and identify which term more data reduces.",
        "Show how the softmax function is invariant to a constant shift and why that makes the log-sum-exp form stable.",
    ],
    "algorithms": [
        "Explain amortized analysis for a dynamic array, using the accounting and potential methods.",
        "Compare Dijkstra with an index heap and Bellman-Ford for a graph with a few negative edges, and justify your choice.",
        "Explain a suffix automaton and when it is preferable to a suffix array plus LCP for substring queries.",
        "Describe a fast algorithm for computing the k-th smallest element and prove its expected linear time bound.",
        "Explain segment trees with lazy propagation and a case where lazy propagation is subtly wrong if order is mishandled.",
        "Discuss the difference between approximation ratio and practical performance for the travelling salesman problem.",
        "Explain topological sorting, cycle detection, and how a scheduler uses both to detect a deadlocked dependency graph.",
        "Describe a union-find with path compression and union by rank, and derive its amortized complexity.",
        "Explain a rolling hash for substring matching and how to make the collision probability negligible.",
        "Describe a sweep-line algorithm for rectangle intersection and the data structure it needs.",
        "Explain how a treap maintains balance by random priority and what that implies for expected depth.",
        "Describe a bitset-backed algorithm for set intersection on thousands of small integer sets.",
    ],
    "ml_moe": [
        "Explain mixture-of-experts routing, load balancing, and why auxiliary balancing losses can hurt quality.",
        "Describe top-k expert selection with normalized gate weights, and explain how the normalization affects layer output scale.",
        "Explain why expert storage locality dominates MoE inference cost on a memory-constrained machine.",
        "Compare dense scaling with sparsely activated MoE for inference on a device with 16 GB of unified memory.",
        "Explain grouped-query attention and how it reduces KV cache memory without changing output shape.",
        "Discuss how a router's temperature or jitter noise changes expert assignment diversity during training.",
        "Explain how an expert-parallel MoE layer communicates during training and where the bandwidth bottleneck is.",
        "Describe how a MoE layer's capacity factor causes token dropping and what that does to a training step.",
        "Explain how a shared expert in a fine-grained MoE architecture changes the routing statistics.",
        "Discuss how quantization error interacts with rarely-used experts and why some experts are more sensitive.",
        "Explain how the choice of top-k during inference changes output distribution relative to the full-expert mixture.",
        "Describe an evaluation protocol for asking whether a pruned expert set still produces the same outputs.",
    ],
    "inference_opt": [
        "Explain KV caching, its memory growth with context length, and two ways to reduce the footprint without retraining.",
        "Describe speculative decoding and explain the conditions under which it actually improves tokens per second.",
        "Explain how weight quantization to four bits interacts with outlier channels and what a mixed-precision fallback buys you.",
        "Discuss kernel fusion opportunities in a transformer block and the point at which fusion stops helping.",
        "Explain why batching improves GPU utilization but can hurt latency, and how continuous batching changes the tradeoff.",
        "Describe how a prefetcher for expert weights interacts with an SSD bottlenecked by random reads.",
        "Explain the difference between compute-bound and memory-bandwidth-bound transformer layers, and how to tell which you are in.",
        "Describe how to measure tokens per second honestly when model loading and prefill are part of the user-visible latency.",
        "Explain how a paged KV cache lets requests share memory and where fragmentation still appears.",
        "Discuss how to decide between a smaller model at higher precision and a larger quantization of a bigger model.",
        "Explain how an attention kernel's memory traffic scales with sequence length and batch size.",
        "Describe how to profile a decoder to separate host overhead from device execution time.",
    ],
    "prose": [
        "Explain to a non-specialist why a computer needs cache hierarchies rather than just more RAM.",
        "Write a short technical essay about why prediction quality and the cost of acting on a prediction should be judged separately.",
        "Explain in plain language what happens between pressing a key and a character appearing in a terminal.",
        "Write a short explanation of the CAP theorem and one common misreading of it.",
        "Explain the difference between latency and throughput for a storage device, using a queueing argument.",
        "Describe how to reason about a system's p99 latency when you can only measure the mean.",
        "Explain how a garbage collector's generational hypothesis changes application pause behaviour.",
        "Describe how a virtual machine implements a dynamically typed language's inline caches.",
        "Explain in plain terms what a hash function is used for and why collisions are unavoidable.",
        "Describe what an operating system actually does when a program asks for a large block of memory.",
        "Explain why two programs that produce the same output can differ in correctness.",
        "Write a short explanation of why floating-point addition is not associative.",
    ],
    "debug_review": [
        "Give a structured debugging plan for a program that produces the wrong answer only once every few thousand runs.",
        "You are reviewing a production inference engine. List the highest-risk correctness bugs around caching, cancellation, and thread synchronization.",
        "Write a careful code review of a function that mixes integer and floating-point arithmetic in a loop bound.",
        "Explain how to bisect a performance regression with unreliable microbenchmarks and noisy hosting.",
        "Write a detailed plan for benchmarking an optimization when host thermals and filesystem cache can create misleading speedups.",
        "Explain why a benchmark that repeats a query in a loop can report a speedup that does not exist in production.",
        "Describe how to make a numerical kernel's test suite catch a transposed index that only matters for non-square inputs.",
        "Write unit tests that would catch an off-by-one in a ring buffer's full-versus-empty distinction.",
        "Explain how to review a patch that changes a memory allocator, and what invariants you would test first.",
        "Describe a strategy for reproducing a bug that depends on thread scheduling order.",
        "Explain how a fuzzer can discover a parser bug that a handwritten test suite missed, and what corpus you would seed it with.",
        "Describe how to review a change to a cache eviction policy for correctness and not just hit rate.",
    ],
    "systems_extra": [
        "Explain how a GPU executes a warp with divergent branches and why divergence costs throughput.",
        "Describe a SIMD implementation of a dot product and the alignment and tail-handling issues you must handle.",
        "Explain how a floating-point format's exponent bias affects the range of representable values.",
        "Compare a binary heap and a pairing heap for workloads with decrease-key operations.",
        "Explain how a text editor implements undo with a persistent data structure and how it bounds memory.",
        "Describe how an image codec uses a discrete cosine transform and quantization in a lossy pipeline.",
        "Explain how a compiler's linker resolves symbols and what happens when two object files define the same weak symbol.",
        "Explain how an operating system scheduler balances latency-sensitive and throughput-oriented threads.",
        "Describe how a secure enclave or hardware key store protects a secret during use.",
        "Explain how a filesystem journal recovers after a crash and why the ordering of metadata writes matters.",
        "Compare a token-bucket and a leaky-bucket rate limiter for a shared downstream service.",
        "Explain how a load balancer chooses a backend when latency measurements arrive asynchronously and are stale.",
        "Describe how to shard a growing key-value store while keeping the migration invisible to clients.",
        "Explain how a versioned key-value store implements a read-modify-write with optimistic concurrency.",
        "Explain what happens inside a CPU when a speculative load misses in the cache and the speculation is squashed.",
        "Describe a branch predictor that learns from history and explain how a misprediction costs a pipeline flush.",
        "Explain how a memory allocator interacts with a thread-local cache and why that reduces contention.",
        "Explain how a compiler's escape analysis lets it allocate on the stack instead of the heap.",
        "Describe how a type checker handles inference for a generic function with a recursive constraint.",
        "Explain how a tracing JIT chooses a hot loop and what triggers it to fall back to the interpreter.",
    ],
}



def _round_robin(groups: dict[str, list[str]]) -> list[str]:
    """Interleave domains so every prefix spans all of them."""
    ordered = sorted(groups.items())
    out: list[str] = []
    depth = max(len(v) for _, v in ordered)
    for i in range(depth):
        for _name, items in ordered:
            if i < len(items):
                out.append(items[i])
    return out


TRAIN_BANK = _round_robin(_DOMAINS)

# --- validation (model selection only) ---------------------------------

VAL_BANK = [
    "Explain Rust's trait coherence rules and why an orphan-rule violation cannot be fixed by adding a local wrapper type without a newtype.",
    "Implement a C++ lock-free stack with hazard pointers and explain the reclamation race you must close.",
    "Explain how a compiler turns a switch statement into a jump table and when it prefers a decision tree instead.",
    "Describe how a database's write-ahead log interacts with a group commit policy during a burst of small transactions.",
    "Explain how a distributed consensus protocol behaves when the leader is partitioned but still believes it leads.",
    "Derive the update rule for gradient descent on a logistic regression loss and explain the role of the learning rate.",
    "Explain how an inference engine decides to evict a cached expert block and what metrics drive that decision.",
    "Write a structured plan for diagnosing a memory leak that only appears after several hours of steady load.",
]

# --- final unseen evaluation -------------------------------------------

TEST_BANK = [
    "Explain how Rust's async runtime parks and wakes tasks, and how a poorly written future can starve an executor thread.",
    "Explain how a C compiler's optimizer can remove an entire loop whose body has no observable effect, and when volatile prevents it.",
    "Describe how a relational optimizer estimates join cardinality and where its estimates fail badly.",
    "Explain how NVMe completion interrupts versus polling affect throughput and CPU cost for a single reader.",
    "Explain the difference between a distributed lock and a lease, and how clock skew breaks the lease assumption.",
    "Explain how a sparse MoE layer's output changes when the gate's top-k normalization is applied over a truncated expert set.",
    "Write a proof that a randomised quickselect has linear expected running time and identify where the expectation is taken.",
    "Explain how a storage engine's compaction scheduler trades write amplification against read amplification.",
]

ALL_BANKS = {"train": TRAIN_BANK, "val": VAL_BANK, "test": TEST_BANK}

if __name__ == "__main__":
    for name, bank in ALL_BANKS.items():
        print(f"{name}: {len(bank)} prompts, {len(set(bank))} unique")
    print("train head (round-robin check):")
    for i, prompt in enumerate(TRAIN_BANK[:8]):
        print(f"  {i}: {prompt[:72]}")
