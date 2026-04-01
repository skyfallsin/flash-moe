// test_ane_ssd_overlap.cpp — Test whether ANE compute overlaps with SSD pread()
//
// The core hypothesis: ANE has a separate DMA/memory path from the GPU.
// GPU compute + SSD pread contend on the shared memory controller.
// If ANE compute + SSD pread DON'T contend, we can overlap them.
//
// Experiment:
//   1. Measure ANE matmul latency alone (many iterations)
//   2. Measure SSD pread latency alone (many iterations)  
//   3. Measure ANE matmul + SSD pread overlapped (concurrent)
//   4. If overlapped time ~ max(ANE, SSD), they don't contend
//      If overlapped time ~ ANE + SSD, they fully contend

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <dispatch/dispatch.h>
#include "core/ane_runtime.h"

static double now_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

// ============================================================================
// SSD benchmark helpers
// ============================================================================

static const char* BENCH_FILE = "/tmp/ane_ssd_bench.dat";
static const size_t BENCH_FILE_SIZE = 256 * 1024 * 1024; // 256 MB
static const size_t PREAD_SIZE = 5 * 1024 * 1024;        // 5 MB per read (~ 1 expert)

static void create_bench_file() {
    if (access(BENCH_FILE, F_OK) == 0) {
        struct stat st;
        stat(BENCH_FILE, &st);
        if ((size_t)st.st_size >= BENCH_FILE_SIZE) return;
    }
    printf("Creating %zu MB benchmark file...\n", BENCH_FILE_SIZE / (1024*1024));
    int fd = open(BENCH_FILE, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) { perror("create bench file"); exit(1); }
    
    // Write random data to prevent compression
    std::vector<uint8_t> buf(1024 * 1024);
    for (size_t i = 0; i < buf.size(); i++) buf[i] = (uint8_t)(rand() & 0xFF);
    for (size_t written = 0; written < BENCH_FILE_SIZE; written += buf.size()) {
        write(fd, buf.data(), buf.size());
    }
    close(fd);
    
    // Purge from page cache
    printf("Purging page cache...\n");
    system("sudo purge 2>/dev/null || true");
}

// Warm the file into page cache
static void warm_bench_file(int fd) {
    char* buf = (char*)malloc(PREAD_SIZE);
    for (size_t off = 0; off < BENCH_FILE_SIZE; off += PREAD_SIZE) {
        pread(fd, buf, PREAD_SIZE, off);
    }
    free(buf);
}

struct PreadResult {
    double elapsed_us;
    size_t bytes_read;
};

// Read K experts worth of data from random offsets
static PreadResult do_pread_batch(int fd, int num_reads) {
    size_t total_bytes = 0;
    std::vector<char> buf(PREAD_SIZE);
    
    double t0 = now_us();
    for (int i = 0; i < num_reads; i++) {
        off_t offset = (rand() % ((BENCH_FILE_SIZE - PREAD_SIZE) / 4096)) * 4096;
        ssize_t n = pread(fd, buf.data(), PREAD_SIZE, offset);
        if (n > 0) total_bytes += n;
    }
    double t1 = now_us();
    return {t1 - t0, total_bytes};
}

// ============================================================================
// ANE benchmark helpers
// ============================================================================

struct ANEBenchKernels {
    ane_lm::ANEKernel* matmul_4096_12288;  // QKV proj (largest)
    ane_lm::ANEKernel* matmul_4096_8192;   // Z proj / O proj
    ane_lm::ANEKernel* matmul_4096_4096;   // mid-size
    ane_lm::ANEKernel* fused_ffn;          // shared expert FFN
};

static ANEBenchKernels compile_bench_kernels() {
    ANEBenchKernels k = {};
    
    // Generate random BF16 weights
    auto make_weights = [](int rows, int cols) -> std::vector<uint16_t> {
        std::vector<uint16_t> w(rows * cols);
        for (size_t i = 0; i < w.size(); i++) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            w[i] = f32_to_bf16(v);
        }
        return w;
    };
    
    printf("Compiling ANE kernels (this takes ~30-60s)...\n");
    
    // 4096 -> 12288 (QKV projection, largest dense op)
    {
        auto w = make_weights(12288, 4096);
        printf("  Compiling 4096->12288 matmul...\n");
        k.matmul_4096_12288 = ane_lm::ane_compile_matmul(w.data(), 12288, 4096);
        if (!k.matmul_4096_12288) printf("  FAILED to compile 4096->12288!\n");
        else printf("  OK\n");
    }
    
    // 4096 -> 8192 (Z projection)
    {
        auto w = make_weights(8192, 4096);
        printf("  Compiling 4096->8192 matmul...\n");
        k.matmul_4096_8192 = ane_lm::ane_compile_matmul(w.data(), 8192, 4096);
        if (!k.matmul_4096_8192) printf("  FAILED to compile 4096->8192!\n");
        else printf("  OK\n");
    }
    
    // 4096 -> 4096
    {
        auto w = make_weights(4096, 4096);
        printf("  Compiling 4096->4096 matmul...\n");
        k.matmul_4096_4096 = ane_lm::ane_compile_matmul(w.data(), 4096, 4096);
        if (!k.matmul_4096_4096) printf("  FAILED to compile 4096->4096!\n");
        else printf("  OK\n");
    }
    
    // Shared expert FFN: 4096 -> 1024 -> 4096
    {
        auto gate_w = make_weights(1024, 4096);
        auto up_w = make_weights(1024, 4096);
        auto down_w = make_weights(4096, 1024);
        printf("  Compiling fused FFN 4096->1024->4096...\n");
        k.fused_ffn = ane_lm::ane_compile_fused_ffn(gate_w.data(), up_w.data(), down_w.data(), 4096, 1024);
        if (!k.fused_ffn) printf("  FAILED to compile fused FFN!\n");
        else printf("  OK\n");
    }
    
    return k;
}

static double bench_ane_matmul(ane_lm::ANEKernel* k, int in_dim, int out_dim, int iters) {
    std::vector<float> input(in_dim, 0.1f);
    std::vector<float> output(out_dim);
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        ane_lm::ane_matvec(k, output.data(), input.data(), in_dim, out_dim);
    }
    
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        ane_lm::ane_matvec(k, output.data(), input.data(), in_dim, out_dim);
    }
    double t1 = now_us();
    return (t1 - t0) / iters;
}

static double bench_ane_ffn(ane_lm::ANEKernel* k, int dim, int inter, int iters) {
    std::vector<float> input(dim, 0.1f);
    std::vector<float> output(dim);
    
    for (int i = 0; i < 5; i++) {
        ane_lm::ane_matvec(k, output.data(), input.data(), dim, dim);
    }
    
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        ane_lm::ane_matvec(k, output.data(), input.data(), dim, dim);
    }
    double t1 = now_us();
    return (t1 - t0) / iters;
}

// ============================================================================
// Overlap test: ANE + SSD concurrent
// ============================================================================

struct OverlapArgs {
    ane_lm::ANEKernel* kernel;
    int in_dim;
    int out_dim;
    int ane_iters;
    double ane_elapsed_us;
    
    int fd;
    int pread_count;
    double pread_elapsed_us;
    size_t pread_bytes;
};

static void* ane_thread_func(void* arg) {
    OverlapArgs* a = (OverlapArgs*)arg;
    std::vector<float> input(a->in_dim, 0.1f);
    std::vector<float> output(a->out_dim);
    
    double t0 = now_us();
    for (int i = 0; i < a->ane_iters; i++) {
        ane_lm::ane_matvec(a->kernel, output.data(), input.data(), a->in_dim, a->out_dim);
    }
    a->ane_elapsed_us = now_us() - t0;
    return NULL;
}

static void* pread_thread_func(void* arg) {
    OverlapArgs* a = (OverlapArgs*)arg;
    std::vector<char> buf(PREAD_SIZE);
    size_t total = 0;
    
    double t0 = now_us();
    for (int i = 0; i < a->pread_count; i++) {
        off_t offset = (rand() % ((BENCH_FILE_SIZE - PREAD_SIZE) / 4096)) * 4096;
        ssize_t n = pread(a->fd, buf.data(), PREAD_SIZE, offset);
        if (n > 0) total += n;
    }
    a->pread_elapsed_us = now_us() - t0;
    a->pread_bytes = total;
    return NULL;
}

struct OverlapResult {
    double ane_alone_us;
    double pread_alone_us;
    double overlapped_wall_us;
    double ane_during_overlap_us;
    double pread_during_overlap_us;
    double overlap_ratio; // 1.0 = perfect overlap, 0.0 = full contention
};

static OverlapResult run_overlap_test(
    ane_lm::ANEKernel* kernel, int in_dim, int out_dim,
    int fd, int ane_iters, int pread_count, int trials)
{
    std::vector<float> input(in_dim, 0.1f);
    std::vector<float> output(out_dim);
    
    // Measure ANE alone
    double ane_total = 0;
    for (int t = 0; t < trials; t++) {
        for (int i = 0; i < 3; i++) ane_lm::ane_matvec(kernel, output.data(), input.data(), in_dim, out_dim);
        double t0 = now_us();
        for (int i = 0; i < ane_iters; i++) {
            ane_lm::ane_matvec(kernel, output.data(), input.data(), in_dim, out_dim);
        }
        ane_total += now_us() - t0;
    }
    double ane_alone = ane_total / trials;
    
    // Measure pread alone
    double pread_total = 0;
    for (int t = 0; t < trials; t++) {
        auto r = do_pread_batch(fd, pread_count);
        pread_total += r.elapsed_us;
    }
    double pread_alone = pread_total / trials;
    
    // Measure overlapped
    double wall_total = 0, ane_overlap_total = 0, pread_overlap_total = 0;
    for (int t = 0; t < trials; t++) {
        // Warmup ANE
        for (int i = 0; i < 3; i++) ane_lm::ane_matvec(kernel, output.data(), input.data(), in_dim, out_dim);
        
        OverlapArgs args = {};
        args.kernel = kernel;
        args.in_dim = in_dim;
        args.out_dim = out_dim;
        args.ane_iters = ane_iters;
        args.fd = fd;
        args.pread_count = pread_count;
        
        pthread_t ane_tid, pread_tid;
        double wall_t0 = now_us();
        pthread_create(&ane_tid, NULL, ane_thread_func, &args);
        pthread_create(&pread_tid, NULL, pread_thread_func, &args);
        pthread_join(ane_tid, NULL);
        pthread_join(pread_tid, NULL);
        double wall_t1 = now_us();
        
        wall_total += (wall_t1 - wall_t0);
        ane_overlap_total += args.ane_elapsed_us;
        pread_overlap_total += args.pread_elapsed_us;
    }
    
    OverlapResult r;
    r.ane_alone_us = ane_alone;
    r.pread_alone_us = pread_alone;
    r.overlapped_wall_us = wall_total / trials;
    r.ane_during_overlap_us = ane_overlap_total / trials;
    r.pread_during_overlap_us = pread_overlap_total / trials;
    
    // overlap_ratio: how much of the shorter task was "free"
    // If wall ~ max(ane, pread), ratio = 1 (perfect overlap)
    // If wall ~ ane + pread, ratio = 0 (no overlap)
    double expected_serial = r.ane_alone_us + r.pread_alone_us;
    double expected_overlap = std::max(r.ane_alone_us, r.pread_alone_us);
    if (expected_serial > expected_overlap) {
        r.overlap_ratio = (expected_serial - r.overlapped_wall_us) / (expected_serial - expected_overlap);
    } else {
        r.overlap_ratio = 0;
    }
    
    return r;
}

// ============================================================================
// GPU benchmark for comparison (Metal command buffer)
// ============================================================================

// We'll use a simple cblas_sgemm as a proxy for "GPU-like memory bus usage"
// since the actual GPU contention test needs Metal setup
#include <Accelerate/Accelerate.h>

static double bench_cpu_gemm(int M, int K, int iters) {
    // M x K matmul (simulates a matvec with batch)
    std::vector<float> A(M * K, 0.01f);
    std::vector<float> x(K, 0.1f);
    std::vector<float> y(M);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K, 1.0f,
                    A.data(), K, x.data(), 1, 0.0f, y.data(), 1);
    }
    
    double t0 = now_us();
    for (int i = 0; i < iters; i++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K, 1.0f,
                    A.data(), K, x.data(), 1, 0.0f, y.data(), 1);
    }
    return (now_us() - t0) / iters;
}

// CPU GEMV + pread overlap (as comparison baseline)
struct CPUOverlapArgs {
    int M, K, iters;
    double elapsed_us;
    int fd;
    int pread_count;
    double pread_elapsed_us;
    size_t pread_bytes;
};

static void* cpu_gemv_thread(void* arg) {
    CPUOverlapArgs* a = (CPUOverlapArgs*)arg;
    std::vector<float> A(a->M * a->K, 0.01f);
    std::vector<float> x(a->K, 0.1f);
    std::vector<float> y(a->M);
    
    double t0 = now_us();
    for (int i = 0; i < a->iters; i++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, a->M, a->K, 1.0f,
                    A.data(), a->K, x.data(), 1, 0.0f, y.data(), 1);
    }
    a->elapsed_us = now_us() - t0;
    return NULL;
}

static void* cpu_pread_thread(void* arg) {
    CPUOverlapArgs* a = (CPUOverlapArgs*)arg;
    std::vector<char> buf(PREAD_SIZE);
    size_t total = 0;
    double t0 = now_us();
    for (int i = 0; i < a->pread_count; i++) {
        off_t offset = (rand() % ((BENCH_FILE_SIZE - PREAD_SIZE) / 4096)) * 4096;
        ssize_t n = pread(a->fd, buf.data(), PREAD_SIZE, offset);
        if (n > 0) total += n;
    }
    a->pread_elapsed_us = now_us() - t0;
    a->pread_bytes = total;
    return NULL;
}

// ============================================================================
// Main
// ============================================================================

static void print_overlap_result(const char* label, const OverlapResult& r) {
    printf("\n=== %s ===\n", label);
    printf("  ANE alone:       %8.1f us\n", r.ane_alone_us);
    printf("  pread alone:     %8.1f us\n", r.pread_alone_us);
    printf("  Overlapped wall: %8.1f us\n", r.overlapped_wall_us);
    printf("  ANE in overlap:  %8.1f us (%.1f%% slowdown)\n",
           r.ane_during_overlap_us,
           100.0 * (r.ane_during_overlap_us - r.ane_alone_us) / r.ane_alone_us);
    printf("  pread in overlap:%8.1f us (%.1f%% slowdown)\n",
           r.pread_during_overlap_us,
           100.0 * (r.pread_during_overlap_us - r.pread_alone_us) / r.pread_alone_us);
    printf("  Expected serial: %8.1f us\n", r.ane_alone_us + r.pread_alone_us);
    printf("  Expected overlap:%8.1f us\n", std::max(r.ane_alone_us, r.pread_alone_us));
    printf("  Overlap ratio:   %.1f%% ", r.overlap_ratio * 100.0);
    if (r.overlap_ratio > 0.8) printf("(EXCELLENT — near-perfect overlap)\n");
    else if (r.overlap_ratio > 0.5) printf("(GOOD — significant overlap)\n");
    else if (r.overlap_ratio > 0.2) printf("(PARTIAL — some contention)\n");
    else printf("(POOR — heavy contention)\n");
}

int main(int argc, char** argv) {
    printf("=== ANE vs SSD Overlap Benchmark ===\n");
    printf("Testing whether ANE compute can overlap with SSD pread()\n\n");
    
    int trials = 5;
    int ane_iters = 50;
    int pread_count = 4;  // K=4 experts
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trials") == 0 && i+1 < argc) trials = atoi(argv[++i]);
        if (strcmp(argv[i], "--ane-iters") == 0 && i+1 < argc) ane_iters = atoi(argv[++i]);
        if (strcmp(argv[i], "--preads") == 0 && i+1 < argc) pread_count = atoi(argv[++i]);
    }
    
    printf("Config: trials=%d, ane_iters=%d, pread_count=%d, pread_size=%.1f MB\n",
           trials, ane_iters, pread_count, PREAD_SIZE / (1024.0*1024.0));
    
    // 1. Create benchmark file
    create_bench_file();
    int fd = open(BENCH_FILE, O_RDONLY);
    if (fd < 0) { perror("open bench file"); return 1; }
    
    // Warm file into page cache (simulates flash-moe's "trust the OS" approach)
    printf("Warming file into page cache...\n");
    warm_bench_file(fd);
    
    // 2. Initialize ANE
    printf("\nInitializing ANE...\n");
    ane_lm::ane_init();
    if (!ane_lm::ane_available()) {
        printf("ANE not available! Cannot run overlap test.\n");
        close(fd);
        return 1;
    }
    printf("ANE available.\n\n");
    
    // 3. Compile ANE kernels at 397B dimensions
    ANEBenchKernels kernels = compile_bench_kernels();
    
    printf("\n--- Individual Benchmarks ---\n");
    
    // 4. Benchmark ANE alone
    if (kernels.matmul_4096_12288) {
        double us = bench_ane_matmul(kernels.matmul_4096_12288, 4096, 12288, 100);
        printf("ANE 4096->12288 matvec: %.1f us/iter\n", us);
    }
    if (kernels.matmul_4096_8192) {
        double us = bench_ane_matmul(kernels.matmul_4096_8192, 4096, 8192, 100);
        printf("ANE 4096->8192 matvec:  %.1f us/iter\n", us);
    }
    if (kernels.matmul_4096_4096) {
        double us = bench_ane_matmul(kernels.matmul_4096_4096, 4096, 4096, 100);
        printf("ANE 4096->4096 matvec:  %.1f us/iter\n", us);
    }
    
    // 5. Benchmark pread alone
    {
        double total = 0;
        for (int t = 0; t < 10; t++) {
            auto r = do_pread_batch(fd, pread_count);
            total += r.elapsed_us;
        }
        printf("pread %dx%.1fMB (warm cache): %.1f us/batch\n",
               pread_count, PREAD_SIZE/(1024.0*1024.0), total/10.0);
    }
    
    // 6. CPU GEMV + pread overlap (baseline comparison)
    printf("\n--- CPU GEMV + pread Overlap (baseline) ---\n");
    {
        int M = 12288, K = 4096;
        double cpu_alone = bench_cpu_gemm(M, K, ane_iters);
        printf("CPU GEMV %dx%d: %.1f us/iter, %d iters = %.1f us\n",
               M, K, cpu_alone, ane_iters, cpu_alone * ane_iters);
        
        CPUOverlapArgs cargs = {};
        cargs.M = M; cargs.K = K; cargs.iters = ane_iters;
        cargs.fd = fd; cargs.pread_count = pread_count;
        
        // Measure alone
        double cpu_batch = cpu_alone * ane_iters;
        auto pr = do_pread_batch(fd, pread_count);
        
        // Measure overlapped
        double wall_total = 0;
        for (int t = 0; t < trials; t++) {
            cargs.elapsed_us = 0;
            cargs.pread_elapsed_us = 0;
            pthread_t t1, t2;
            double wt0 = now_us();
            pthread_create(&t1, NULL, cpu_gemv_thread, &cargs);
            pthread_create(&t2, NULL, cpu_pread_thread, &cargs);
            pthread_join(t1, NULL);
            pthread_join(t2, NULL);
            wall_total += now_us() - wt0;
        }
        double wall_avg = wall_total / trials;
        printf("  CPU alone:   %.1f us\n", cpu_batch);
        printf("  pread alone: %.1f us\n", pr.elapsed_us);
        printf("  Overlapped:  %.1f us\n", wall_avg);
        double serial = cpu_batch + pr.elapsed_us;
        double best = std::max(cpu_batch, pr.elapsed_us);
        double ratio = (serial > best) ? (serial - wall_avg) / (serial - best) : 0;
        printf("  Overlap ratio: %.1f%%\n", ratio * 100);
    }
    
    // 7. ANE + pread overlap tests
    printf("\n--- ANE + pread Overlap Tests ---\n");
    
    if (kernels.matmul_4096_12288) {
        auto r = run_overlap_test(kernels.matmul_4096_12288, 4096, 12288,
                                  fd, ane_iters, pread_count, trials);
        print_overlap_result("ANE QKV (4096->12288) + pread K=4 experts", r);
    }
    
    if (kernels.matmul_4096_8192) {
        auto r = run_overlap_test(kernels.matmul_4096_8192, 4096, 8192,
                                  fd, ane_iters, pread_count, trials);
        print_overlap_result("ANE Z-proj (4096->8192) + pread K=4 experts", r);
    }
    
    // Test with more pread to simulate full layer I/O
    if (kernels.matmul_4096_12288) {
        auto r = run_overlap_test(kernels.matmul_4096_12288, 4096, 12288,
                                  fd, ane_iters, pread_count * 2, trials);
        print_overlap_result("ANE QKV + pread K=8 experts (2x I/O)", r);
    }
    
    // Test FFN + pread
    if (kernels.fused_ffn) {
        auto r = run_overlap_test(kernels.fused_ffn, 4096, 4096,
                                  fd, ane_iters, pread_count, trials);
        print_overlap_result("ANE shared FFN (4096->1024->4096) + pread K=4", r);
    }
    
    printf("\n=== Summary ===\n");
    printf("If overlap ratios are >80%%, ANE+SSD can be pipelined profitably.\n");
    printf("If <20%%, ANE shares the memory controller bottleneck with SSD DMA.\n");
    printf("flash-moe reports GPU+SSD CANNOT overlap (shared memory controller).\n");
    printf("The question is whether ANE's dedicated DMA path is different.\n");
    
    // Cleanup
    if (kernels.matmul_4096_12288) ane_lm::ane_free(kernels.matmul_4096_12288);
    if (kernels.matmul_4096_8192) ane_lm::ane_free(kernels.matmul_4096_8192);
    if (kernels.matmul_4096_4096) ane_lm::ane_free(kernels.matmul_4096_4096);
    if (kernels.fused_ffn) ane_lm::ane_free(kernels.fused_ffn);
    close(fd);
    
    return 0;
}
