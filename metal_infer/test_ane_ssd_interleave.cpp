// Interleaved measurement: compare ANE latency with/without concurrent pread
// More robust than batch overlap since it measures per-call impact
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
#include <atomic>
#include "core/ane_runtime.h"

static double now_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16);
}

static const char* BENCH_FILE = "/tmp/ane_ssd_bench.dat";
static const size_t FILE_SIZE = 256 * 1024 * 1024;
static const size_t READ_SIZE = 5 * 1024 * 1024;

static std::atomic<bool> g_io_running{false};
static std::atomic<bool> g_stop_io{false};
static std::atomic<uint64_t> g_io_bytes{0};

static void* continuous_io_thread(void* arg) {
    int fd = *(int*)arg;
    void* buf;
    posix_memalign(&buf, 4096, READ_SIZE);
    g_io_running = true;
    uint64_t total = 0;
    while (!g_stop_io.load(std::memory_order_relaxed)) {
        off_t offset = ((off_t)(rand() % ((FILE_SIZE - READ_SIZE) / 4096))) * 4096;
        ssize_t n = pread(fd, buf, READ_SIZE, offset);
        if (n > 0) total += n;
    }
    g_io_bytes = total;
    free(buf);
    return NULL;
}

int main() {
    printf("=== ANE Latency: Alone vs During Concurrent SSD I/O ===\n\n");
    
    ane_lm::ane_init();
    if (!ane_lm::ane_available()) { printf("No ANE\n"); return 1; }
    
    int in_dim = 4096, out_dim = 12288;
    std::vector<uint16_t> w(out_dim * in_dim);
    for (size_t i = 0; i < w.size(); i++) w[i] = f32_to_bf16(((float)(rand()%1000)/1000 - 0.5f) * 0.1f);
    printf("Compiling ANE 4096->12288...\n");
    auto* k = ane_lm::ane_compile_matmul(w.data(), out_dim, in_dim);
    if (!k) { printf("FAILED\n"); return 1; }
    printf("OK\n");
    
    std::vector<float> input(in_dim, 0.1f), output(out_dim);
    // Warmup ANE
    for (int i = 0; i < 30; i++) ane_lm::ane_matvec(k, output.data(), input.data(), in_dim, out_dim);
    
    int N = 200;  // number of ANE calls to measure
    
    // Phase 1: ANE alone
    printf("\nPhase 1: %d ANE calls, no I/O...\n", N);
    std::vector<double> alone_lats;
    for (int i = 0; i < N; i++) {
        double t0 = now_us();
        ane_lm::ane_matvec(k, output.data(), input.data(), in_dim, out_dim);
        alone_lats.push_back(now_us() - t0);
    }
    
    std::sort(alone_lats.begin(), alone_lats.end());
    double alone_p50 = alone_lats[N/2];
    double alone_mean = std::accumulate(alone_lats.begin(), alone_lats.end(), 0.0) / N;
    printf("  ANE alone: p50=%.0f us, mean=%.0f us, p10=%.0f, p90=%.0f\n",
           alone_p50, alone_mean, alone_lats[N/10], alone_lats[N*9/10]);
    
    // Phase 2: ANE with concurrent warm-cache I/O
    printf("\nPhase 2: %d ANE calls + continuous warm-cache pread...\n", N);
    {
        int fd = open(BENCH_FILE, O_RDONLY);
        // Warm cache
        char* wbuf = (char*)malloc(READ_SIZE);
        for (size_t off = 0; off < FILE_SIZE; off += READ_SIZE) pread(fd, wbuf, READ_SIZE, off);
        free(wbuf);
        
        g_stop_io = false; g_io_running = false; g_io_bytes = 0;
        pthread_t io_tid;
        pthread_create(&io_tid, NULL, continuous_io_thread, &fd);
        while (!g_io_running) usleep(100);
        usleep(1000); // let I/O thread get going
        
        std::vector<double> warm_lats;
        for (int i = 0; i < N; i++) {
            double t0 = now_us();
            ane_lm::ane_matvec(k, output.data(), input.data(), in_dim, out_dim);
            warm_lats.push_back(now_us() - t0);
        }
        
        g_stop_io = true;
        pthread_join(io_tid, NULL);
        
        std::sort(warm_lats.begin(), warm_lats.end());
        double warm_p50 = warm_lats[N/2];
        double warm_mean = std::accumulate(warm_lats.begin(), warm_lats.end(), 0.0) / N;
        printf("  ANE+warmIO: p50=%.0f us, mean=%.0f us, p10=%.0f, p90=%.0f\n",
               warm_p50, warm_mean, warm_lats[N/10], warm_lats[N*9/10]);
        printf("  IO throughput: %.1f GB/s (warm cache)\n",
               g_io_bytes.load() / 1e9 / (warm_mean * N / 1e6));
        printf("  p50 impact: %+.1f%% (%s)\n", 100*(warm_p50 - alone_p50)/alone_p50,
               fabs(warm_p50 - alone_p50)/alone_p50 < 0.05 ? "NO CONTENTION" :
               (warm_p50 - alone_p50)/alone_p50 < 0.15 ? "MILD" : "SIGNIFICANT");
        close(fd);
    }
    
    // Phase 3: ANE with concurrent F_NOCACHE I/O (real SSD)
    printf("\nPhase 3: %d ANE calls + continuous F_NOCACHE pread (real SSD)...\n", N);
    {
        int fd = open(BENCH_FILE, O_RDONLY);
        fcntl(fd, F_NOCACHE, 1);
        
        g_stop_io = false; g_io_running = false; g_io_bytes = 0;
        pthread_t io_tid;
        pthread_create(&io_tid, NULL, continuous_io_thread, &fd);
        while (!g_io_running) usleep(100);
        usleep(1000);
        
        std::vector<double> cold_lats;
        for (int i = 0; i < N; i++) {
            double t0 = now_us();
            ane_lm::ane_matvec(k, output.data(), input.data(), in_dim, out_dim);
            cold_lats.push_back(now_us() - t0);
        }
        
        g_stop_io = true;
        pthread_join(io_tid, NULL);
        
        std::sort(cold_lats.begin(), cold_lats.end());
        double cold_p50 = cold_lats[N/2];
        double cold_mean = std::accumulate(cold_lats.begin(), cold_lats.end(), 0.0) / N;
        printf("  ANE+SSD:    p50=%.0f us, mean=%.0f us, p10=%.0f, p90=%.0f\n",
               cold_p50, cold_mean, cold_lats[N/10], cold_lats[N*9/10]);
        printf("  IO throughput: %.1f GB/s (F_NOCACHE)\n",
               g_io_bytes.load() / 1e9 / (cold_mean * N / 1e6));
        printf("  p50 impact: %+.1f%% vs alone (%s)\n", 100*(cold_p50 - alone_p50)/alone_p50,
               fabs(cold_p50 - alone_p50)/alone_p50 < 0.05 ? "NO CONTENTION" :
               (cold_p50 - alone_p50)/alone_p50 < 0.15 ? "MILD" : "SIGNIFICANT");
        close(fd);
    }
    
    printf("\n=== Verdict ===\n");
    printf("If ANE p50 stays within ~5%% during I/O, the memory paths are independent.\n");
    printf("This would validate the ANE+SSD overlap hypothesis for flash-moe.\n");
    
    ane_lm::ane_free(k);
    return 0;
}
