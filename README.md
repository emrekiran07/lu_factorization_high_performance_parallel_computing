1 Introduction
LU factorization is a fundamental method used in linear algebra computations to
solve matrices and is widely used in various scientific and engineering applications.
When working with large matrices, the computational time can increase significantly
since the computational complexity is O(N3).This creates a big performance
bottleneck, especially in systems that work on large datasets.
Traditional (serial) LU factorization cannot fully utilize the parallel processing
power offered by multi-core processors because it involves sequential computation
steps. Therefore, it is important to accelerate this process by using parallel programming
techniques.

In this study, parallelization of LU factorization using OpenMP is being worked
on. First, basic OpenMP parallelization techniques are applied, and then performance
is increased by various optimizations (workload balancing, memory access
regulation, synchronization, block factorization etc).
A parallel version is developed using OpenMP, and the effects of different strategies
(e.g., blocking and lock mechanisms) are examined.
In the experimental analysis, performance measurements performed at different
matrix sizes are presented, and the speedups obtained are evaluated.

2 Problem Description
LU factorization is the process of decomposing a square matrix as the product of
two triangular matrices. This decomposition, expressed in the form
A = L × U
consists of L (lower triangular matrix) and U (upper triangular matrix) components.
Traditional LU factorization has O(N3) time complexity and can become inefficient
due to its high computational cost, especially for large matrices. Serial
(single-threaded) implementation can not use processor resources efficiently and is
not suitable for parallel processing. Therefore, it is necessary to parallelize LU factorization
using OpenMP on multi-core processors. But there are some problems
occur with this. Dependencies, since there is column dependency in LU factorization,
some operations must be completed first and then other operations must
be performed. Memory access pattern, cache performance of LU factorization is
poor. Memory access pattern needs to be improved. Synchronization and Data
Race, multiple threads accessing the same data at the same time increases the risk
of producing erroneous results. To prevent this, locks or critical sections should
be used. Within this project’s scope, serial and parallel LU factorizations were
compared and the best performance was tried to be obtained by using different
OpenMP optimizations.

3 Solution Method
OpenMP Parallelization, I parallelized the classical LU factorization algorithm
using OpenMP. Inner loops are parallelized with the #pragma omp parallel for
structure. Different schedule strategies (static, dynamic, dynamic with chunk size)
were tried and the most efficient one was selected.
Improving workload distribution, dynamic scheduling was implemented using chunk
size so that threads could share the load equally.
Safe synchronization with lock mechanism, in some cases, different threads may
try to access the same data. This may cause a data race condition. Access to
critical regions is synchronized by defining omp_lock_t type locks.
Parallelization of k-loop, the k loop which is the outermost loop in the LU factorization
operates in series, which limits the overall parallel speedup. In the k cycle,
I tried to reduce the bottleneck by using column-wise partitioning and OpenMP
tasks wherever possible.

Data locality optimization, blocking technique was applied to increase cache performance while accessing matrix elements. Matrices were divided into small blocks
and each block was processed independently.
Block based LU Factorization (Block LU), in the latest version of the algorithm,
block-based LU factorization was applied instead of classical LU. With this method,
the matrix was divided into submatrices and, the LU operation was applied on each
block. Higher cache efficiency and better parallelization were aimed, especially for
large-sized matrices.

4 Experiments
In this section, I compare the parallel and serial versions of our LU decomposition
algorithm to investigate the performance impact of different OpenMP settings.
Experiments were executed with -O0 and -O3 flags to observe the effect of compiler
optimization. Matrices of size 512 and 1000 were tested with block sizes of 64 and
128.

To ensure correctness, the computed L and U matrices were multiplied, and the resulting
matrix was compared to the original input matrix A. A numerical tolerance
(ϵ = 1e−2) was used to account for floating-point rounding errors.
This section presents a comparative performance analysis between the Baseline
and Version 1 (V1) of the LU factorization implementation, both developed using
OpenMP. The Baseline represents a naive parallel version with separate parallel
regions inside loops, while V1 introduces critical improvements: a single persistent
parallel region and proper barrier synchronization to reduce thread management
overhead and avoid race conditions.

All versions were compiled using gcc with -O0 and -O3 optimization flags. Experiments
were conducted on matrix sizes N = 1000, 2000, 3000, 4000, and 5000.
Timing was measured using omp_get_wtime() for algorithm runtime. No blocking
(B) is used in the Baseline and V1 versions. All computations proceed on the
original matrix in row-major or column-major layout (depending on implementation).
The Baseline version uses separate OpenMP regions within each loop body. While
simple, this approach suffers from frequent thread creation/destruction within each
loop iteration, high thread management overhead, and race conditions due to the
lack of proper barriers or synchronization. 

Version 1 introduces structural parallel
optimizations such as a single parallel region encompassing the outer loop and
manual barriers to control synchronization between stages. This version preserved
logical LU operation order with synchronization to prevent data risks.
To evaluate the impact of optimization in Version 1, we compare it directly against
the baseline implementation for a large matrix size of N=5000, using the -O3 compiler optimization flag to allow the compiler to produce highly optimized code.
Total LU time was reduced by ∼ 1.3 seconds (∼ 9%) in V1. System time dropped
by nearly 78%, from 4.047 s to 0.909 s. The user time also saw a modest reduction,
indicating more efficient CPU utilization. These results clearly show that restructuring
the parallel region in LU factorization reduces synchronization overhead
and improves overall execution efficiency, even before introducing more advanced
optimization techniques like blocking or layout transformations.

Version 2 introduces a memory locality optimization using "first-touch" allocation.
In this version, all matrix memory (A, L, U) is initialized in a parallel region so
that the memory pages are associated with the threads that will use them. This
improves performance by taking advantage of memory access patterns and reducing
cache misses. Version 1 performance observed as 0.0555s as well as Version
2 improved to 0.0362s for N=1000. Version 2 reduces real execution time by approximately
35%, mainly due to improved memory locality and cache behavior.
This shows that even without algorithmic changes, memory access patterns have
a measurable impact on performance at scale.

Version 3 is a fine-grained lock-based parallel LU factorization. In this version, I introduced
OpenMP locks to implement fine-grained synchronization during LU factorization.
While previous versions relied solely on barriers to prevent race conditions,
this version uses a per-column lock array to ensure that updates to L[i][k] are
completed before threads read from it to update U[i][j]. By using omp_set_lock
and omp_unset_lock, each thread safely coordinates access to shared columns,
allowing greater concurrency and reducing unnecessary synchronization overhead.
Additionally, column-wise initialization was parallelized, and threads were assigned
slices of columns to balance the workload.
In performance testing with matrix size N=1000, Version 3 achieved a LU computation
time of 1.612 seconds, compared to 0.036 seconds in Version 2. While
this version introduces more robust synchronization, the locking overhead led to
increased execution time. However, the result is valuable in contexts where correctness
under parallel execution must be guaranteed, especially when race conditions
might otherwise arise.

Version 4 introduces crucial optimizations that significantly improve performance
and parallel efficiency over Version 3. In V3, individual column locks were used to
ensure data consistency, which introduced synchronization overhead. V4 replaces
these with OpenMP barriers, which are lighter-weight and eliminate contention
around critical sections. Previously, only a single thread was responsible for computing
all values in column k of matrix L. V4 distributes this computation across
all threads using #pragma omp for, enabling full CPU utilization and reducing idle
time. In C, matrices are stored in row-major order. V3 accessed matrix U columnwise,
leading to inefficient memory usage. V4 fixes this by updating U row-wise,
improving spatial locality and cache performance. This version significantly outperforms V3, achieving over 24× speedup in LU Time (1.612 s → 0.038 s), thanks
to better parallel resource usage and optimized memory behavior. It demonstrates
how small structural adjustments in synchronization and data access patterns can
yield substantial improvements in performance on large matrix problems.

In Version 5, the LU factorization logic preserving its columnwise structure and
synchronization semantics. I introduced per-column locks (omp_lock_t) to guard
dependencies and enforced a strict execution order across threads using finegrained
locking. However, this came at a significant performance cost. Due to
excessive synchronization and single-thread bottlenecks for computing parts of the
matrix (e.g., L[i][k] only being computed by thread 0), parallel efficiency dropped
substantially. This inefficiency is largely due to over-synchronization through locks
and barriers, limited parallelism in critical sections, and poor memory locality from
column-major style access in a row-major C environment.

To improve cache locality and performance, I implemented a blocked LU factorization
approach, dividing the matrix into sub-blocks of size B ×B and operating
on these blocks sequentially. In the baseline blocked LU version, all computations
are performed serially. The diagonal block is factorized, followed by updates to the
row panel (U), column panel (L), and trailing submatrix. This version achieved a
runtime of 0.096970 seconds for matrix size N = 512, block size B = 64.
I then parallelized the panel and submatrix updates using #pragma omp parallel
for, creating the Blocked LU Parallel version. By parallelizing steps 2–4 (row
panel, column panel, and trailing submatrix updates), the workload is distributed
across multiple threads, substantially accelerating execution. This parallel version
completed in 0.019890 seconds, yielding a 4.8× speedup over the serial baseline.
This significant performance improvement demonstrates the benefits of combining
blocking with OpenMP parallelism, especially when applied to memory-intensive
algorithms like LU decomposition.

To further optimize memory access and parallelism, I implemented a Blocked LU
Column Major version, where matrices are stored and accessed in column-major
order using a flat 1D array. This layout aligns with good memory access and
can improve performance in linear algebra operations when used consistently with
memory access patterns. In addition to tiling the computation using a block size
of B = 64, I applied OpenMP parallelization on the update steps such as the row
panel (U), the column panel (L), and the trailing submatrix update. This version
achieved a runtime of 0.028859 seconds for N = 512, which is approximately
3.3× faster than the serial blocked baseline (0.096970 s), though it was slightly
slower than the row-major parallel version (0.019890 s). This minor slowdown is
attributed to less optimal memory locality on C-based row-major architectures,
where column-major traversal can introduce more cache misses.

5 Conclusions
In this project, we explored and optimized LU factorization using a range of parallelization
and memory access strategies. Starting from a simple serial baseline,
we gradually introduced OpenMP parallel regions, synchronization mechanisms,
blocking techniques, memory layout transformations, and compiler optimizations
to improve both performance and scalability. Early optimizations focused on reducing
thread overhead by replacing multiple short-lived parallel regions with a
single persistent one, which led to a noticeable reduction in execution time. Further
enhancements in Version 4 introduced a more efficient memory access pattern
by transitioning to row-major updates, aligning better with the default memory
layout of C, and significantly boosting cache performance. The introduction of
blocked LU algorithms provided additional performance improvements by enabling
better data locality and reducing the volume of repeated memory accesses. Among
these, the parallel blocked LU version using row-major access demonstrated the
highest performance, with execution times significantly lower than any earlier implementations.
On the other hand, the column-major variant, while still correct
and effective, performed slightly worse due to less favorable memory access patterns
on the CPU. The experiments demonstrated that careful attention to thread
usage, synchronization strategy, and data layout can lead to significant improvements
in both runtime and system resource utilization. These results provide a
strong foundation for further improvements and potential extensions to distributed
systems or GPU acceleration in future work.
