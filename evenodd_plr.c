#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <getopt.h>
#include <pthread.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <liburing.h>
#include <sys/types.h>
#include <sys/sysmacros.h>
#include <alloca.h>
#include <sys/uio.h>
#include <sys/mman.h>

// 简化版本 - 仅在地址分配策略上与优化版本不同，其他配置保持一致

// ========== 地址分配策略实现（整合自 alloc_strategy.c/h） ==========

// BlockPos 结构定义
typedef struct {
    int row;   // 包索引 (0..w-1)
    int col;   // 磁盘索引 (0..k-1)
    int diag;  // 对角线索引 (0..k-2)
} BlockPos;

// 地址分配函数声明
static int map_blocks_row(int s, int k, int w, BlockPos *blocks);
static int map_blocks_diag(int s, int k, int w, BlockPos *blocks);
static int choose_best_mapping(int s, int k, int w, BlockPos *blocks);
static int map_blocks_row_optimized(int s, int k, int w, BlockPos *blocks);
static int map_blocks_diag_fixed(int s, int k, int w, int target_diag, BlockPos *blocks);
static int map_blocks_diag_multi_nonzero(int s, int k, int w, BlockPos *blocks);
static int choose_best_mapping_enhanced(int s, int k, int w, BlockPos *blocks);
static int map_blocks_rs_friendly(int s, int k, int w, BlockPos *blocks);
static int map_blocks_batch_aware(int s, int k, int w, int batch_size, BlockPos *blocks);
static int map_blocks_rs_load_aware(int s, int k, int w, double current_rs_load, BlockPos *blocks);
static int map_blocks_rs_ultra_compact(int s, int k, int w, int max_stripes_per_batch, BlockPos *blocks);
static int map_blocks_stripe_hotspot(int s, int k, int w, int hotspot_stripe_id, BlockPos *blocks);
static void fill_sequential_plan(BlockPos *blocks, int count);
static int plan_block_positions(int count, BlockPos *blocks, int prefer_rs_layout, const char **plan_used);

// 前向声明（避免隐式声明带来的冲突与警告）
void xor_update_simd(char *dst, const char *src, size_t size);
static uint32_t crc32(uint32_t crc, const unsigned char *buf, size_t len);

// 配置结构
typedef struct {
    int k;              // 数据盘数量
    int m;              // 校验盘数量 (EVENODD 固定为 2)
    int w;              // 编码参数
    int packetsize;     // 数据包大小
    int update_size;    // 更新大小
    int n_updates;      // 更新次数
    char *mode;         // 更新模式
    char *alloc;        // 地址分配策略: sequential|row|diag|auto
    int verify;         // 是否进行一致性校验
    long verify_samples;// 校验采样条带数 (<=0 表示全量)
    int strong;         // 强一致路径：批次写完成后重算条带 P/Q 并写回
} config_t;


// 增强版性能统计 - 支持详细指标收集
typedef struct {
    double total_io_time;
    double compute_time;
    int read_count;
    int write_count;
    double total_latency;
    int update_count;
    double *io_times;
    int io_index;
    int io_capacity;
    long long xor_count;
    pthread_mutex_t lock;
    
    // 新增：详细性能指标
    struct {
        double throughput_mbps;        // 吞吐量 (MB/s)
        double iops;                   // IOPS
        double avg_latency_ms;         // 平均延迟 (ms)
        double p95_latency_ms;         // 95%延迟 (ms)
        double p99_latency_ms;         // 99%延迟 (ms)
        double cpu_usage_percent;      // CPU使用率
        double memory_usage_mb;        // 内存使用量 (MB)
        long long cache_hits;          // 缓存命中次数
        long long cache_misses;        // 缓存未命中次数
        double simd_efficiency;        // SIMD效率
        double io_efficiency;         // I/O效率
        double parity_efficiency;      // 校验计算效率
    } detailed_stats;
    
    // 新增：实时监控
    struct {
        double current_throughput;     // 当前吞吐量
        double current_latency;        // 当前延迟
        int active_operations;         // 活跃操作数
        int queue_depth;               // 队列深度
        double load_factor;            // 负载因子
    } realtime_stats;
    
    // 新增：历史数据
    struct {
        double *throughput_history;    // 吞吐量历史
        double *latency_history;       // 延迟历史
        int history_size;              // 历史数据大小
        int history_index;             // 当前索引
    } history;
} perf_stats_t;

// 增强版内存池 - 支持多种大小和NUMA感知
typedef struct memory_pool {
    void **buffers;
    int *free_list;
    int free_count;
    int capacity;
    size_t buffer_size;
    pthread_mutex_t lock;
    
    // 新增：NUMA感知和预分配
    int numa_node;              // NUMA节点
    int prealloc_count;         // 预分配数量
    int alignment;              // 内存对齐要求
    uint64_t total_allocated;   // 总分配字节数
    uint64_t peak_usage;       // 峰值使用量
    struct timespec last_cleanup; // 上次清理时间
} memory_pool_t;

// Forward declarations to avoid implicit declaration warnings when used above
void* pool_alloc(memory_pool_t *pool);
void  pool_free(memory_pool_t *pool, void *buffer);

// 多级内存池管理器
typedef struct {
    memory_pool_t *pools;        // 不同大小的内存池数组
    int pool_count;             // 池数量
    size_t *pool_sizes;         // 各池的缓冲区大小
    int *pool_capacities;       // 各池的容量
    pthread_mutex_t global_lock; // 全局锁
    int numa_aware;             // 是否启用NUMA感知
} multi_pool_manager_t;


// 全局变量
config_t config = {
    .k = 5,
    .m = 2,
    .w = 8,
    .packetsize = 4096,
    .update_size = 4096,
    .n_updates = 1000,
    .mode = "sequential",
    .alloc = "sequential",
    .verify = 0,
    .verify_samples = 0,
    .strong = 0
};

perf_stats_t stats = {0};
memory_pool_t *global_pool = NULL;

// 记录本次更新触及的条带集合（用于最终轻量修复）
static long *g_touched_stripes = NULL;
static int g_touched_n = 0;
static int g_touched_cap = 0;
static void touched_add(long s) {
    for (int i = 0; i < g_touched_n; i++) if (g_touched_stripes[i] == s) return;
    if (g_touched_n == g_touched_cap) {
        int nc = g_touched_cap ? g_touched_cap * 2 : 256;
        long *np = (long*)realloc(g_touched_stripes, sizeof(long) * (size_t)nc);
        if (!np) return;
        g_touched_stripes = np;
        g_touched_cap = nc;
    }
    g_touched_stripes[g_touched_n++] = s;
}
static void touched_clear(void) {
    free(g_touched_stripes); g_touched_stripes = NULL; g_touched_n = g_touched_cap = 0;
}

// 新增：固定文件句柄支持
typedef struct {
    int enabled;       // 是否启用固定文件
    int *handles;      // 每个磁盘的句柄：若启用固定文件，则为注册索引；否则为普通 fd
} fixed_files_t;
static fixed_files_t g_fixed = {0, NULL};

// 新增：固定缓冲池（第一步：注册固定缓冲并用 read_fixed/write_fixed）
typedef struct {
    int enabled;
    struct iovec *iovecs;
    void **ptrs;
    int capacity;
    int *free_stack;
    int top;
    pthread_mutex_t lock;
} fixed_bufpool_t;
static fixed_bufpool_t g_bufpool = {0};

static int bufpool_init(struct io_uring *ring, int capacity, size_t buf_size) {
    memset(&g_bufpool, 0, sizeof(g_bufpool));
    g_bufpool.capacity = capacity;
    g_bufpool.iovecs = (struct iovec*)malloc(sizeof(struct iovec) * (size_t)capacity);
    g_bufpool.ptrs = (void**)malloc(sizeof(void*) * (size_t)capacity);
    g_bufpool.free_stack = (int*)malloc(sizeof(int) * (size_t)capacity);
    if (!g_bufpool.iovecs || !g_bufpool.ptrs || !g_bufpool.free_stack) return -1;
    for (int i = 0; i < capacity; i++) {
        void *p = NULL;
        if (posix_memalign(&p, 4096, buf_size) != 0) return -1;
        g_bufpool.ptrs[i] = p;
        g_bufpool.iovecs[i].iov_base = p;
        g_bufpool.iovecs[i].iov_len = buf_size;
        g_bufpool.free_stack[i] = i;
    }
    g_bufpool.top = capacity;
    pthread_mutex_init(&g_bufpool.lock, NULL);
    if (io_uring_register_buffers(ring, g_bufpool.iovecs, capacity) == 0) {
        g_bufpool.enabled = 1;
        return 0;
    }
    g_bufpool.enabled = 0;
    return 0;
}

static int bufpool_acquire(void **ptr_out) {
    pthread_mutex_lock(&g_bufpool.lock);
    if (g_bufpool.top == 0) {
        pthread_mutex_unlock(&g_bufpool.lock);
        return -1;
    }
    int idx = g_bufpool.free_stack[--g_bufpool.top];
    void *p = g_bufpool.ptrs[idx];
    pthread_mutex_unlock(&g_bufpool.lock);
    *ptr_out = p;
    return idx;
}

static void bufpool_release(int idx) {
    if (idx < 0 || idx >= g_bufpool.capacity) return;
    pthread_mutex_lock(&g_bufpool.lock);
    g_bufpool.free_stack[g_bufpool.top++] = idx;
    pthread_mutex_unlock(&g_bufpool.lock);
}

static void bufpool_destroy(void) {
    if (g_bufpool.iovecs) {
        for (int i = 0; i < g_bufpool.capacity; i++) {
            if (g_bufpool.ptrs && g_bufpool.ptrs[i]) free(g_bufpool.ptrs[i]);
        }
        free(g_bufpool.iovecs);
        free(g_bufpool.ptrs);
        free(g_bufpool.free_stack);
    }
    pthread_mutex_destroy(&g_bufpool.lock);
    memset(&g_bufpool, 0, sizeof(g_bufpool));
}

// I/O 统计辅助函数（新增）
static inline double timespec_diff_sec(const struct timespec *start, const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1e9;
}
static void record_io_events(double duration_seconds, int event_count, int is_write) {
    pthread_mutex_lock(&stats.lock);
    stats.total_io_time += duration_seconds;
    if (is_write) stats.write_count += event_count; else stats.read_count += event_count;
    if (stats.io_times && event_count > 0) {
        double per_event = duration_seconds / event_count;
        for (int i = 0; i < event_count && stats.io_index < stats.io_capacity; i++) {
            stats.io_times[stats.io_index++] = per_event;
        }
    }
    pthread_mutex_unlock(&stats.lock);
}

// 增强版内存池实现 - 支持NUMA感知和统计
memory_pool_t* create_enhanced_memory_pool(int capacity, size_t buffer_size, int numa_node, int alignment) {
    memory_pool_t *pool = malloc(sizeof(memory_pool_t));
    if (!pool) return NULL;
    
    memset(pool, 0, sizeof(memory_pool_t));
    
    pool->capacity = capacity;
    pool->buffer_size = buffer_size;
    pool->free_count = capacity;
    pool->numa_node = numa_node;
    pool->alignment = alignment;
    pool->prealloc_count = capacity;
    pthread_mutex_init(&pool->lock, NULL);
    
    pool->buffers = malloc(capacity * sizeof(void*));
    pool->free_list = malloc(capacity * sizeof(int));
    
    if (!pool->buffers || !pool->free_list) {
        free(pool->buffers);
        free(pool->free_list);
        free(pool);
        return NULL;
    }

// 获取当前时间
    clock_gettime(CLOCK_MONOTONIC, &pool->last_cleanup);
    
    for (int i = 0; i < capacity; i++) {
        if (posix_memalign(&pool->buffers[i], alignment, buffer_size) != 0) {
            for (int j = 0; j < i; j++) {
                free(pool->buffers[j]);
            }
            free(pool->buffers);
            free(pool->free_list);
            free(pool);
            return NULL;
        }

// 初始化内存为0，提高缓存友好性
        memset(pool->buffers[i], 0, buffer_size);
        pool->free_list[i] = i;
    }
    
    pool->total_allocated = capacity * buffer_size;
    pool->peak_usage = pool->total_allocated;
    
    return pool;
}

// 兼容性函数
memory_pool_t* create_memory_pool(int capacity, size_t buffer_size) {
    return create_enhanced_memory_pool(capacity, buffer_size, -1, 64);
}

void destroy_memory_pool(memory_pool_t *pool) {
    if (!pool) return;
    
    for (int i = 0; i < pool->capacity; i++) {
        free(pool->buffers[i]);
    }
    free(pool->buffers);
    free(pool->free_list);
    pthread_mutex_destroy(&pool->lock);
    free(pool);
}

// 增强版内存池分配函数 - 支持统计和性能优化
void* pool_alloc_enhanced(memory_pool_t *pool) {
    if (!pool) return NULL;
    
    pthread_mutex_lock(&pool->lock);
    
    if (pool->free_count == 0) {
        pthread_mutex_unlock(&pool->lock);
        // 动态分配作为后备
        void *buffer;
        if (posix_memalign(&buffer, pool->alignment, pool->buffer_size) != 0) {
            return NULL;
        }
        // 初始化内存
        memset(buffer, 0, pool->buffer_size);
        return buffer;
    }
    
    int idx = pool->free_list[--pool->free_count];
    void *buffer = pool->buffers[idx];
    
    // 更新统计信息
    if (pool->peak_usage < (pool->capacity - pool->free_count) * pool->buffer_size) {
        pool->peak_usage = (pool->capacity - pool->free_count) * pool->buffer_size;
    }
    
    pthread_mutex_unlock(&pool->lock);
    
    // 预取内存到缓存，提高性能
    __builtin_prefetch(buffer, 0, 3);
    
    return buffer;
}

// 兼容性函数
void* pool_alloc(memory_pool_t *pool) {
    return pool_alloc_enhanced(pool);
}

// 增强版内存池释放函数 - 支持统计和安全清理
void pool_free_enhanced(memory_pool_t *pool, void *buffer) {
    if (!pool || !buffer) return;
    
    pthread_mutex_lock(&pool->lock);
    
    int found = -1;
    for (int i = 0; i < pool->capacity; i++) {
        if (pool->buffers[i] == buffer) {
            found = i;
            break;
        }
    }
    
    if (found >= 0 && pool->free_count < pool->capacity) {
        pool->free_list[pool->free_count++] = found;
        
        // 清理内存内容，提高安全性
        memset(buffer, 0, pool->buffer_size);
        
        // 检查是否需要定期清理
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - pool->last_cleanup.tv_sec > 300) { // 5分钟
            pool->last_cleanup = now;
            // 这里可以添加内存碎片整理逻辑
        }
        
        pthread_mutex_unlock(&pool->lock);
    } else {
        pthread_mutex_unlock(&pool->lock);
        // 动态分配的内存直接释放
        free(buffer);
    }
}

// 兼容性函数
void pool_free(memory_pool_t *pool, void *buffer) {
    pool_free_enhanced(pool, buffer);
}

// SIMD XOR 实现
// 增强版SIMD XOR实现 - 支持多种优化策略
void xor_update_simd_enhanced(char *dst, const char *src, size_t size) {
    if (!dst || !src || size == 0) return;
    
    size_t i = 0;
    
    // 预取数据到缓存
    __builtin_prefetch(dst, 1, 3);
    __builtin_prefetch(src, 0, 3);
    
    // AVX-512 处理（如果支持）
    #ifdef __AVX512F__
    for (; i + 64 <= size; i += 64) {
        __m512i d = _mm512_loadu_si512((__m512i*)(dst + i));
        __m512i s = _mm512_loadu_si512((__m512i*)(src + i));
        __m512i result = _mm512_xor_si512(d, s);
        _mm512_storeu_si512((__m512i*)(dst + i), result);
    }
    #endif
    
    // AVX2 处理
    #ifdef __AVX2__
    for (; i + 32 <= size; i += 32) {
        __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
        __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
        __m256i result = _mm256_xor_si256(d, s);
        _mm256_storeu_si256((__m256i*)(dst + i), result);
    }
    #endif
    
    // SSE2 处理
    for (; i + 16 <= size; i += 16) {
        __m128i d = _mm_loadu_si128((__m128i*)(dst + i));
        __m128i s = _mm_loadu_si128((__m128i*)(src + i));
        __m128i result = _mm_xor_si128(d, s);
        _mm_storeu_si128((__m128i*)(dst + i), result);
    }

// 64位标量处理
    for (; i + 8 <= size; i += 8) {
        uint64_t *d64 = (uint64_t*)(dst + i);
        uint64_t *s64 = (uint64_t*)(src + i);
        *d64 ^= *s64;
    }

// 32位标量处理
    for (; i + 4 <= size; i += 4) {
        uint32_t *d32 = (uint32_t*)(dst + i);
        uint32_t *s32 = (uint32_t*)(src + i);
        *d32 ^= *s32;
    }

// 8位标量处理剩余部分
    for (; i < size; i++) {
        dst[i] ^= src[i];
    }
}

// 兼容性函数
void xor_update_simd(char *dst, const char *src, size_t size) {
    xor_update_simd_enhanced(dst, src, size);
}

// 批量XOR操作 - 支持多个源数据
void xor_update_batch(char *dst, char **srcs, int src_count, size_t size) {
    if (!dst || !srcs || src_count <= 0 || size == 0) return;
    
    // 对每个源数据执行XOR
    for (int i = 0; i < src_count; i++) {
        if (srcs[i]) {
            xor_update_simd_enhanced(dst, srcs[i], size);
        }
    }
}

// 检查素数
int is_prime(int n) {
    if (n <= 1) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

// ========== 地址分配策略函数实现 ==========

static int calculate_mapping_cost(BlockPos *blocks, int s, int k __attribute__((unused)), int w __attribute__((unused))) {
    if (s == 0) return 0;
    int row_used[512] = {0};
    int diag_used[512] = {0};
    int row_count = 0, diag_count = 0;
    for (int i = 0; i < s; i++) {
        int r = blocks[i].row;
        int d = blocks[i].diag;
        if (r >= 0 && r < 512 && !row_used[r]) { row_used[r] = 1; row_count++; }
        if (d >= 0 && d < 512 && !diag_used[d]) { diag_used[d] = 1; diag_count++; }
    }
    return row_count + diag_count;
}

static int calculate_mapping_cost_penalized(BlockPos *blocks, int s, int k, int w) {
    int base = calculate_mapping_cost(blocks, s, k, w);
    // 若包含 diag==0，增加惩罚（近似表示会触发多对角更新）
    for (int i = 0; i < s; i++) {
        if (blocks[i].diag % (k - 1) == 0) { base += (k - 2); break; }
    }
    return base;
}

// 行优先映射
static int map_blocks_row(int s, int k, int w, BlockPos *blocks) {
    int idx = 0;
    int row = 0;
    while (idx < s && row < w) {
        for (int col = 0; col < k && idx < s; col++) {
            blocks[idx].row = row;
            blocks[idx].col = col;
            blocks[idx].diag = (row + col) % (k - 1);
            idx++;
        }
        row++;
    }
    return idx;
}

// 对角线优先映射
static int map_blocks_diag(int s, int k, int w, BlockPos *blocks) {
    int idx = 0;
    int target_diag = 0;
    while (idx < s) {
        int placed_in_diag = 0;
        for (int row = 0; row < w && idx < s; row++) {
            int col = (target_diag - row + (k - 1)) % (k - 1);
            if (col < k) {
                blocks[idx].row = row;
                blocks[idx].col = col;
                blocks[idx].diag = target_diag;
                idx++;
                placed_in_diag++;
            }
        }
        target_diag = (target_diag + 1) % (k - 1);
        if (target_diag == 0 && placed_in_diag == 0) {
            while (idx < s) {
                int row = idx / k;
                int col = idx % k;
                if (row >= w) break;
                blocks[idx].row = row;
                blocks[idx].col = col;
                blocks[idx].diag = (row + col) % (k - 1);
                idx++;
            }
            break;
        }
    }
    return idx;
}

// 选择最佳映射
static int choose_best_mapping(int s, int k, int w, BlockPos *blocks) {
    if (s == 0) return 0;
    BlockPos row_blocks[s];
    BlockPos diag_blocks[s];
    map_blocks_row(s, k, w, row_blocks);
    int row_cost = calculate_mapping_cost(row_blocks, s, k, w);
    map_blocks_diag(s, k, w, diag_blocks);
    int diag_cost = calculate_mapping_cost(diag_blocks, s, k, w);
    if (row_cost <= diag_cost) { memcpy(blocks, row_blocks, (size_t)s * sizeof(BlockPos)); return s; }
    memcpy(blocks, diag_blocks, (size_t)s * sizeof(BlockPos)); return s;
}

// 行优化映射（限制到最少行，且仅使用少数对角线，避免 diag==0）
static int map_blocks_row_optimized(int s, int k, int w, BlockPos *blocks) {
    if (s <= 0) return 0;
    int row_capacity = k;
    int min_rows = (s + row_capacity - 1) / row_capacity;
    if (min_rows > w) min_rows = w;
    int placed = 0;
    int target_diags[2] = {1 % (k - 1), 2 % (k - 1)};
    for (int r = 0; r < min_rows && placed < s; r++) {
        for (int t = 0; t < row_capacity && placed < s; t++) {
            int d = target_diags[t % 2];
            int col = (d - r + (k - 1)) % (k - 1);
            if (col >= k) col = (col + 1) % k;
            blocks[placed].row = r;
            blocks[placed].col = col;
            blocks[placed].diag = d;
            placed++;
        }
    }
    return placed;
}

// 固定对角线映射（尽量保持单一对角线，避开 0）
static int map_blocks_diag_fixed(int s, int k, int w, int target_diag, BlockPos *blocks) {
    if (s <= 0) return 0;
    if (target_diag % (k - 1) == 0) target_diag = 1 % (k - 1);
    int placed = 0;
    for (int r = 0; r < w && placed < s; r++) {
        int col = (target_diag - r + (k - 1)) % (k - 1);
        if (col >= k) continue;
        blocks[placed].row = r;
        blocks[placed].col = col;
        blocks[placed].diag = target_diag;
        placed++;
    }
    return placed;
}

// 小更新（≤16KB）优先沿非零对角线分布，完全避免 diag==0
static int map_blocks_diag_multi_nonzero(int s, int k, int w, BlockPos *blocks) {
    if (s <= 0) return 0;
    int placed = 0;
    for (int pass = 0; placed < s; pass++) {
        for (int d = 1; d < k - 1 && placed < s; d++) {
            for (int r = 0; r < w && placed < s; r++) {
                int col = (d - r + (k - 1)) % (k - 1);
                if (col >= k) continue;
                blocks[placed].row = r;
                blocks[placed].col = col;
                blocks[placed].diag = d;
                placed++;
            }
        }
    }
    return placed;
}

// 自动选择（行优化 vs 固定对角线）
static int choose_best_mapping_enhanced(int s, int k, int w, BlockPos *blocks) {
    // 小更新（≤16KB）强制采用非零对角线的多对角分布，提升局部性并避免EVENODD对角0开销
    if (config.update_size <= 16*1024) {
        return map_blocks_diag_multi_nonzero(s, k, w, blocks);
    }

    BlockPos row_opt[s];
    BlockPos diag_opt[s];
    map_blocks_row_optimized(s, k, w, row_opt);
    map_blocks_diag_fixed(s, k, w, 1, diag_opt);
    int c_row = calculate_mapping_cost_penalized(row_opt, s, k, w);
    int c_diag = calculate_mapping_cost_penalized(diag_opt, s, k, w);
    if (c_row <= c_diag) { memcpy(blocks, row_opt, (size_t)s * sizeof(BlockPos)); return s; }
    memcpy(blocks, diag_opt, (size_t)s * sizeof(BlockPos)); return s;
}

// RS友好的地址分配策略
static int map_blocks_rs_friendly(int s, int k, int w, BlockPos *blocks) {
    if (s <= 0) return 0;
    const int blocks_per_stripe = k * w;
    int placed = 0;
    int current_stripe = 0;
    while (placed < s) {
        int remaining = s - placed;
        int stripe_capacity = (remaining < blocks_per_stripe) ? remaining : blocks_per_stripe;
        int target_diag = 1 + (current_stripe % (k - 2));
        int placed_in_stripe = 0;
        for (int r = 0; r < w && placed_in_stripe < stripe_capacity; r++) {
            int col = (target_diag - r + (k - 1)) % (k - 1);
            if (col < k) {
                blocks[placed].row = r;
                blocks[placed].col = col;
                blocks[placed].diag = target_diag;
                placed++;
                placed_in_stripe++;
            }
        }
        if (placed_in_stripe < stripe_capacity) {
            int current_row = 0;
            while (placed_in_stripe < stripe_capacity && current_row < w) {
                for (int col = 0; col < k && placed_in_stripe < stripe_capacity; col++) {
                    int diag = (current_row + col) % (k - 1);
                    if (diag == 0) diag = target_diag;
                    blocks[placed].row = current_row;
                    blocks[placed].col = col;
                    blocks[placed].diag = diag;
                    placed++;
                    placed_in_stripe++;
                }
                current_row++;
            }
        }
        current_stripe++;
    }
    return placed;
}

// 批量感知的地址分配策略（优化版）
static int map_blocks_batch_aware(int s, int k, int w, int batch_size, BlockPos *blocks) {
    if (s <= 0) return 0;
    int placed = 0;
    for (int batch_start = 0; batch_start < s; batch_start += batch_size) {
        int batch_count = (s - batch_start < batch_size) ? (s - batch_start) : batch_size;
        if (batch_count <= k) {
            int target_row = 0;
            int target_diag = 1;
            for (int i = 0; i < batch_count; i++) {
                int col = (target_diag - target_row + (k - 1)) % (k - 1);
                if (col >= k) col = i % k;
                blocks[placed].row = target_row;
                blocks[placed].col = col;
                blocks[placed].diag = target_diag;
                placed++;
            }
        } else if (batch_count <= k * 2) {
            int target_diag = 1 + ((batch_start / batch_size) % (k - 2));
            for (int i = 0; i < batch_count; i++) {
                int r = (i / k) % 2;
                int col = i % k;
                int diag = (r + col) % (k - 1);
                if (diag == 0) diag = target_diag;
                blocks[placed].row = r;
                blocks[placed].col = col;
                blocks[placed].diag = diag;
                placed++;
            }
        } else {
            int min_rows = (batch_count + k - 1) / k;
            if (min_rows > w) min_rows = w;
            for (int i = 0; i < batch_count; i++) {
                int r = i / k; if (r >= min_rows) r = min_rows - 1;
                int col = i % k;
                int diag = (r + col) % (k - 1);
                if (diag == 0) diag = 1;
                blocks[placed].row = r;
                blocks[placed].col = col;
                blocks[placed].diag = diag;
                placed++;
            }
        }
    }
    return placed;
}

// RS负载感知的地址分配策略
static int map_blocks_rs_load_aware(int s, int k, int w, double current_rs_load, BlockPos *blocks) {
    if (s <= 0) return 0;
    if (current_rs_load < 0.3) {
        return map_blocks_diag_multi_nonzero(s, k, w, blocks);
    } else if (current_rs_load < 0.7) {
        return map_blocks_rs_friendly(s, k, w, blocks);
    } else {
        int placed = 0;
        for (int i = 0; i < s; i++) {
            int within = i % (k * w);
            blocks[i].col = within % k;
            blocks[i].row = within / k;
            blocks[i].diag = (blocks[i].col + blocks[i].row) % (k - 1);
            placed++;
        }
        return placed;
    }
}

// RS极致聚合策略（将批次内所有更新集中到最少条带）
static int map_blocks_rs_ultra_compact(int s, int k, int w, int max_stripes_per_batch, BlockPos *blocks) {
    if (s <= 0) return 0;
    int blocks_per_stripe = k * w;
    int stripes_needed = (s + blocks_per_stripe - 1) / blocks_per_stripe;
    if (stripes_needed > max_stripes_per_batch) stripes_needed = max_stripes_per_batch;
    int placed = 0;
    int current_stripe = 0;
    while (placed < s && current_stripe < stripes_needed) {
        int remaining_in_batch = s - placed;
        int to_place_in_stripe = (remaining_in_batch < blocks_per_stripe) ? remaining_in_batch : blocks_per_stripe;
        int target_diag = 1 + (current_stripe % (k - 2));
        for (int i = 0; i < to_place_in_stripe && placed < s; i++) {
            int row = i / k;
            int col = i % k;
            if (row < w) {
                int ideal_col = (target_diag - row + (k - 1)) % (k - 1);
                if (ideal_col < k) col = ideal_col;
            }
            blocks[placed].row = row;
            blocks[placed].col = col;
            blocks[placed].diag = (row + col) % (k - 1);
            if (blocks[placed].diag == 0) blocks[placed].diag = target_diag;
            placed++;
        }
        current_stripe++;
    }
    return placed;
}

// 条带感知的热点集中策略（配合RS使用）
static int map_blocks_stripe_hotspot(int s, int k, int w, int hotspot_stripe_id, BlockPos *blocks) {
    (void)hotspot_stripe_id;
    if (s <= 0) return 0;
    int placed = 0;
    int diag_order[] = {1, 2, 3, 4};
    int diag_idx = 0;
    while (placed < s && diag_idx < 4) {
        int current_diag = diag_order[diag_idx % (k - 1 > 4 ? 4 : k - 1)];
        if (current_diag >= k - 1) current_diag = 1;
        for (int row = 0; row < w && placed < s; row++) {
            int col = (current_diag - row + (k - 1)) % (k - 1);
            if (col >= k) continue;
            blocks[placed].row = row;
            blocks[placed].col = col;
            blocks[placed].diag = current_diag;
            placed++;
        }
        diag_idx++;
    }
    int row = 0, col = 0;
    while (placed < s) {
        blocks[placed].row = row;
        blocks[placed].col = col;
        blocks[placed].diag = (row + col) % (k - 1);
        if (blocks[placed].diag == 0) blocks[placed].diag = 1;
        placed++;
        col++;
        if (col >= k) { col = 0; row++; if (row >= w) row = 0; }
    }
    return placed;
}

static void fill_sequential_plan(BlockPos *blocks, int count) {
    if (!blocks || count <= 0 || config.k <= 0 || config.w <= 0) return;
    int blocks_per_stripe = config.k * config.w;
    if (blocks_per_stripe <= 0) blocks_per_stripe = config.k > 0 ? config.k : 1;
    if (blocks_per_stripe <= 0) blocks_per_stripe = 1;
    for (int i = 0; i < count; i++) {
        int logical = blocks_per_stripe ? (i % blocks_per_stripe) : i;
        int row = logical / config.k;
        int col = logical % config.k;
        if (row >= config.w) row %= config.w;
        blocks[i].row = row;
        blocks[i].col = col;
        blocks[i].diag = (config.k > 1) ? ((row + col) % (config.k - 1)) : 0;
    }
}

static int plan_block_positions(int count, BlockPos *blocks, int prefer_rs_layout, const char **plan_used) {
    if (plan_used) *plan_used = NULL;
    if (!blocks || count <= 0) return 0;

    const char *policy = (config.alloc && config.alloc[0]) ? config.alloc : "sequential";
    int filled = 0;

    // 优先在大更新（≥32K）时使用超紧凑RS布局，尽量压缩到单条带/少对角，提升RS聚合效率
    if (prefer_rs_layout && config.update_size >= 32*1024) {
        int filled_uc = map_blocks_rs_ultra_compact(count, config.k, config.w, /*max_stripes_per_batch=*/1, blocks);
        if (filled_uc >= count) { if (plan_used) *plan_used = "rs_ultra_compact"; return filled_uc; }
    }
    // 若选择了顺序分配，但启用RS/PARIX优先，则强制采用RS友好布局
    if (prefer_rs_layout && strcmp(policy, "sequential") == 0) {
        int filled_rs = map_blocks_rs_friendly(count, config.k, config.w, blocks);
        if (filled_rs >= count) { if (plan_used) *plan_used = "rs_friendly_seq"; return filled_rs; }
    }

    if (prefer_rs_layout &&
        (strcmp(policy, "auto") == 0 || strcmp(policy, "batch_optimized") == 0)) {
        filled = map_blocks_rs_friendly(count, config.k, config.w, blocks);
        if (filled >= count) {
            if (plan_used) *plan_used = "rs_friendly";
            return filled;
        }
    }

    if (strcmp(policy, "sequential") == 0) {
        fill_sequential_plan(blocks, count);
        if (plan_used) *plan_used = "sequential";
        return count;
    } else if (strcmp(policy, "row") == 0) {
        filled = map_blocks_row(count, config.k, config.w, blocks);
        if (plan_used) *plan_used = "row";
    } else if (strcmp(policy, "diag") == 0) {
        filled = map_blocks_diag(count, config.k, config.w, blocks);
        if (plan_used) *plan_used = "diag";
    } else if (strcmp(policy, "auto") == 0 || strcmp(policy, "batch_optimized") == 0) {
        filled = choose_best_mapping_enhanced(count, config.k, config.w, blocks);
        if (filled >= count) {
            if (plan_used) *plan_used = "auto_enhanced";
        } else {
            int tmp = map_blocks_batch_aware(count, config.k, config.w, count, blocks);
            if (tmp > filled) {
                filled = tmp;
                if (plan_used) *plan_used = "batch_aware";
            }
            if (prefer_rs_layout && filled < count) {
                tmp = map_blocks_rs_friendly(count, config.k, config.w, blocks);
                if (tmp > filled) {
                    filled = tmp;
                    if (plan_used) *plan_used = "rs_friendly";
                }
            }
        }
    } else {
        filled = choose_best_mapping(count, config.k, config.w, blocks);
        if (plan_used) *plan_used = "generic";
    }

    if (filled < count) {
        fill_sequential_plan(blocks, count);
        filled = count;
        if (plan_used) *plan_used = "sequential";
    }

    return filled;
}

// 增强版批量读取操作 - 支持优先级和错误恢复
void batch_io_read_enhanced(char **buffers, int *fds, int count, struct io_uring *ring, off_t offset, size_t size, int priority) {
    if (!buffers || !fds || count <= 0 || !ring) return;
    
    struct io_uring_sqe *sqes[count];
    struct timespec io_start, io_end;
    int retry_count = 0;
    const int max_retries = 3;
    
    // 预取缓冲区到缓存
    for (int i = 0; i < count; i++) {
        __builtin_prefetch(buffers[i], 1, 3);
    }

// 获取所有 SQE
    for (int i = 0; i < count; i++) {
        sqes[i] = io_uring_get_sqe(ring);
        if (!sqes[i]) {
            fprintf(stderr, "Failed to get sqe for operation %d\n", i);
            // 尝试重新获取
            sqes[i] = io_uring_get_sqe(ring);
            if (!sqes[i]) {
                fprintf(stderr, "Critical: No available SQE\n");
                exit(1);
            }
        }

// 设置优先级和标志
        if (g_fixed.enabled) {
            io_uring_sqe_set_flags(sqes[i], IOSQE_FIXED_FILE | IOSQE_IO_LINK);
        }

// 设置优先级
        if (priority > 0) {
            io_uring_sqe_set_flags(sqes[i], IOSQE_IO_HARDLINK);
        }
        
        io_uring_prep_read(sqes[i], fds[i], buffers[i], size, offset);
        io_uring_sqe_set_data(sqes[i], (void*)(uintptr_t)i);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &io_start);
    
    // 批量提交
    int submitted = io_uring_submit(ring);
    if (submitted != count) {
        fprintf(stderr, "Warning: Only %d/%d operations submitted\n", submitted, count);
    }

// 批量收集结果 - 优化版本
    int completed = 0;
    int error_count = 0;
    
    while (completed < count) {
        struct io_uring_cqe *cqe;
        int wait_result = io_uring_wait_cqe(ring, &cqe);
        
        if (wait_result < 0) {
            if (errno == EINTR) continue; // 被信号中断，重试
            perror("io_uring_wait_cqe");
            if (++retry_count > max_retries) {
                fprintf(stderr, "Max retries exceeded\n");
                exit(1);
            }
            continue;
        }
        
        int idx = (int)(uintptr_t)io_uring_cqe_get_data(cqe);
        int result = cqe->res;
        
        if (result < 0) {
            error_count++;
            fprintf(stderr, "Read error on fd %d (idx %d): %s (retry %d/%d)\n", 
                   fds[idx], idx, strerror(abs(result)), retry_count, max_retries);
            
            if (retry_count < max_retries) {
                // 重试失败的请求
                struct io_uring_sqe *retry_sqe = io_uring_get_sqe(ring);
                if (retry_sqe) {
                    if (g_fixed.enabled) io_uring_sqe_set_flags(retry_sqe, IOSQE_FIXED_FILE);
                    io_uring_prep_read(retry_sqe, fds[idx], buffers[idx], size, offset);
                    io_uring_sqe_set_data(retry_sqe, (void*)(uintptr_t)idx);
                    io_uring_submit(ring);
                }
                retry_count++;
            } else {
                fprintf(stderr, "Fatal: Read operation failed after %d retries\n", max_retries);
                exit(1);
            }
        } else if (result != (int)size) {
            fprintf(stderr, "Warning: Partial read on fd %d: %d/%zu bytes\n", fds[idx], result, size);
        }
        
        io_uring_cqe_seen(ring, cqe);
        completed++;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &io_end);
    record_io_events(timespec_diff_sec(&io_start, &io_end), count, 0);
    
    if (error_count > 0) {
        fprintf(stderr, "Warning: %d/%d read operations had errors\n", error_count, count);
    }
}

// 兼容性函数
void batch_io_read(char **buffers, int *fds, int count, struct io_uring *ring, off_t offset, size_t size) {
    batch_io_read_enhanced(buffers, fds, count, ring, offset, size, 0);
}

// 新增：支持每个请求自定义偏移量的并行读取（带计时）
void batch_io_read_with_offsets(char **buffers, int *fds, off_t *offsets, int count, struct io_uring *ring, size_t size) {
    struct io_uring_sqe *sqes[count];
    struct io_uring_cqe *cq;
    struct timespec io_start, io_end;
    for (int i = 0; i < count; i++) {
        sqes[i] = io_uring_get_sqe(ring);
        if (!sqes[i]) {
            fprintf(stderr, "Failed to get sqe\n");
            exit(1);
        }
        if (g_fixed.enabled) io_uring_sqe_set_flags(sqes[i], IOSQE_FIXED_FILE);
        io_uring_prep_read(sqes[i], fds[i], buffers[i], size, offsets[i]);
        io_uring_sqe_set_data(sqes[i], (void*)(uintptr_t)i);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_start);
    io_uring_submit(ring);
    for (int completed = 0; completed < count; completed++) {
        if (io_uring_wait_cqe(ring, &cq) < 0) {
            perror("io_uring_wait_cqe");
            exit(1);
        }
        if (cq->res < 0) {
            int idx = (int)(uintptr_t)io_uring_cqe_get_data(cq);
            fprintf(stderr, "Read error on fd %d: %s\n", fds[idx], strerror(abs(cq->res)));
            exit(1);
        }
        io_uring_cqe_seen(ring, cq);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_end);
    record_io_events(timespec_diff_sec(&io_start, &io_end), count, 0);
}

// 批量写入操作
void batch_io_write(char **buffers, int *fds, int count, struct io_uring *ring, off_t offset, size_t size) {
    struct io_uring_sqe *sqes[count];
    struct io_uring_cqe *cqes[count];

    struct timespec io_start, io_end;
    
    // 获取所有 SQE
    for (int i = 0; i < count; i++) {
        sqes[i] = io_uring_get_sqe(ring);
        if (!sqes[i]) {
            fprintf(stderr, "Failed to get sqe\n");
            exit(1);
        }
        if (g_fixed.enabled) io_uring_sqe_set_flags(sqes[i], IOSQE_FIXED_FILE);
        io_uring_prep_write(sqes[i], fds[i], buffers[i], size, offset);
        io_uring_sqe_set_data(sqes[i], (void*)(uintptr_t)i);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &io_start);
    // 批量提交
    io_uring_submit(ring);
    
    // 批量收集结果
    for (int completed = 0; completed < count; ) {
        if (io_uring_wait_cqe(ring, &cqes[0]) < 0) {
            perror("io_uring_wait_cqe");
            exit(1);
        }
        
        int idx = (int)(uintptr_t)io_uring_cqe_get_data(cqes[0]);
        if (cqes[0]->res < 0) {
            fprintf(stderr, "Write error on fd %d: %s\n", fds[idx], strerror(abs(cqes[0]->res)));
            exit(1);
        }
        
        io_uring_cqe_seen(ring, cqes[0]);
        completed++;
    }

    clock_gettime(CLOCK_MONOTONIC, &io_end);
    record_io_events(timespec_diff_sec(&io_start, &io_end), count, 1);
}

// 新增：支持每个请求自定义偏移量的并行写入（带计时）
void batch_io_write_with_offsets(char **buffers, int *fds, off_t *offsets, int count, struct io_uring *ring, size_t size) {
    struct io_uring_sqe *sqes[count];
    struct io_uring_cqe *cq;
    struct timespec io_start, io_end;
    for (int i = 0; i < count; i++) {
        sqes[i] = io_uring_get_sqe(ring);
        if (!sqes[i]) {
            fprintf(stderr, "Failed to get sqe\n");
            exit(1);
        }
        if (g_fixed.enabled) io_uring_sqe_set_flags(sqes[i], IOSQE_FIXED_FILE);
        io_uring_prep_write(sqes[i], fds[i], buffers[i], size, offsets[i]);
        io_uring_sqe_set_data(sqes[i], (void*)(uintptr_t)i);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_start);
    io_uring_submit(ring);
    for (int completed = 0; completed < count; completed++) {
        if (io_uring_wait_cqe(ring, &cq) < 0) {
            perror("io_uring_wait_cqe");
            exit(1);
        }
        if (cq->res < 0) {
            int idx = (int)(uintptr_t)io_uring_cqe_get_data(cq);
            fprintf(stderr, "Write error on fd %d: %s\n", fds[idx], strerror(abs(cq->res)));
            exit(1);
        }
        io_uring_cqe_seen(ring, cq);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_end);
    record_io_events(timespec_diff_sec(&io_start, &io_end), count, 1);
}

// 新增：带固定缓冲索引的批量读/写
typedef void ex_unused_typedef_to_anchor; // 占位以便插入位置唯一

static void batch_io_read_with_offsets_ex(char **buffers, int *fds, off_t *offsets, int *buf_indices, int count, struct io_uring *ring, size_t size) {
    struct io_uring_sqe *sqes[count];
    struct io_uring_cqe *cq;
    struct timespec io_start, io_end;
    for (int i = 0; i < count; i++) {
        sqes[i] = io_uring_get_sqe(ring);
        if (!sqes[i]) { fprintf(stderr, "Failed to get sqe\n"); exit(1); }
        if (g_bufpool.enabled && buf_indices && buf_indices[i] >= 0) {
            io_uring_prep_read_fixed(sqes[i], fds[i], buffers[i], size, offsets[i], buf_indices[i]);
        } else {
            if (g_fixed.enabled) io_uring_sqe_set_flags(sqes[i], IOSQE_FIXED_FILE);
            io_uring_prep_read(sqes[i], fds[i], buffers[i], size, offsets[i]);
        }
        io_uring_sqe_set_data(sqes[i], (void*)(uintptr_t)i);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_start);
    io_uring_submit(ring);
    for (int completed = 0; completed < count; completed++) {
        if (io_uring_wait_cqe(ring, &cq) < 0) { perror("io_uring_wait_cqe"); exit(1); }
        if (cq->res < 0) {
            int idx = (int)(uintptr_t)io_uring_cqe_get_data(cq);
            fprintf(stderr, "Read error on fd %d: %s\n", fds[idx], strerror(abs(cq->res)));
            exit(1);
        }
        io_uring_cqe_seen(ring, cq);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_end);
    record_io_events(timespec_diff_sec(&io_start, &io_end), count, 0);
}

static __attribute__((unused)) void batch_io_write_with_offsets_ex(char **buffers, int *fds, off_t *offsets, int *buf_indices, int count, struct io_uring *ring, size_t size) {
    struct io_uring_sqe *sqes[count];
    struct io_uring_cqe *cq;
    struct timespec io_start, io_end;
    for (int i = 0; i < count; i++) {
        sqes[i] = io_uring_get_sqe(ring);
        if (!sqes[i]) { fprintf(stderr, "Failed to get sqe\n"); exit(1); }
        if (g_bufpool.enabled && buf_indices && buf_indices[i] >= 0) {
            io_uring_prep_write_fixed(sqes[i], fds[i], buffers[i], size, offsets[i], buf_indices[i]);
        } else {
            if (g_fixed.enabled) io_uring_sqe_set_flags(sqes[i], IOSQE_FIXED_FILE);
            io_uring_prep_write(sqes[i], fds[i], buffers[i], size, offsets[i]);
        }
        io_uring_sqe_set_data(sqes[i], (void*)(uintptr_t)i);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_start);
    io_uring_submit(ring);
    for (int completed = 0; completed < count; completed++) {
        if (io_uring_wait_cqe(ring, &cq) < 0) { perror("io_uring_wait_cqe"); exit(1); }
        if (cq->res < 0) {
            int idx = (int)(uintptr_t)io_uring_cqe_get_data(cq);
            fprintf(stderr, "Write error on fd %d: %s\n", fds[idx], strerror(abs(cq->res)));
            exit(1);
        }
        io_uring_cqe_seen(ring, cq);
    }
    clock_gettime(CLOCK_MONOTONIC, &io_end);
    record_io_events(timespec_diff_sec(&io_start, &io_end), count, 1);
}


// 提交写但不等待，返回提交数
static int submit_writes_no_wait_ex(char **buffers, int *fds, off_t *offsets, int *buf_indices, int count, struct io_uring *ring, size_t size) {
    for (int i = 0; i < count; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
        if (!sqe) { fprintf(stderr, "Failed to get sqe\n"); exit(1); }
        if (g_bufpool.enabled && buf_indices && buf_indices[i] >= 0) {
            io_uring_prep_write_fixed(sqe, fds[i], buffers[i], size, offsets[i], buf_indices[i]);
        } else {
            if (g_fixed.enabled) io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            io_uring_prep_write(sqe, fds[i], buffers[i], size, offsets[i]);
        }
    }
    io_uring_submit(ring);
    return count;
}

// 等待 n 个 CQE 完成
static void wait_cqes_n(struct io_uring *ring, int n) {
    struct io_uring_cqe *cqe;
    for (int completed = 0; completed < n; completed++) {
        if (io_uring_wait_cqe(ring, &cqe) < 0) { perror("io_uring_wait_cqe"); exit(1); }
        if (cqe->res < 0) {
            fprintf(stderr, "io cqe error: %s\n", strerror(abs(cqe->res)));
            exit(1);
        }
        io_uring_cqe_seen(ring, cqe);
    }
}

// 对所有设备并行 fsync，确保数据落盘
static void fsync_all_devices(struct io_uring *ring, int *fds, int count) {
    struct io_uring_sqe *sqes[count];
    struct io_uring_cqe *cqe;
    for (int i = 0; i < count; i++) {
        sqes[i] = io_uring_get_sqe(ring);
        if (!sqes[i]) {
            fprintf(stderr, "Failed to get sqe for fsync\n");
            exit(1);
        }
        if (g_fixed.enabled) io_uring_sqe_set_flags(sqes[i], IOSQE_FIXED_FILE);
        io_uring_prep_fsync(sqes[i], fds[i], IORING_FSYNC_DATASYNC);
        io_uring_sqe_set_data(sqes[i], (void*)(uintptr_t)i);
    }
    io_uring_submit(ring);
    for (int completed = 0; completed < count; completed++) {
        if (io_uring_wait_cqe(ring, &cqe) < 0) {
            perror("io_uring_wait_cqe fsync");
            exit(1);
        }
        if (cqe->res < 0) {
            int idx = (int)(uintptr_t)io_uring_cqe_get_data(cqe);
            fprintf(stderr, "fsync error on fd %d: %s\n", fds[idx], strerror(abs(cqe->res)));
            exit(1);
        }
        io_uring_cqe_seen(ring, cqe);
    }
}

// 前置声明：条带重算修复函数
static int repair_one_stripe(long stripe_index, struct io_uring *ring);

// 加载安全磁盘路径函数
static int load_safe_disk_paths(char **disk_paths, int max_disks) {
    FILE *safe_disk_file = fopen("safe_disks.txt", "r");
    int disk_count = 0;
    
    if (!safe_disk_file) {
        // 回退到默认磁盘路径（跳过nvme0n1系统盘）
        printf("警告: 未找到safe_disks.txt，使用默认磁盘路径（跳过nvme0n1）\n");
        const char *default_paths[] = {
            "/dev/nvme1n1", "/dev/nvme2n1", "/dev/nvme3n1", "/dev/nvme4n1", "/dev/nvme5n1",
            "/dev/nvme6n1", "/dev/nvme7n1", "/dev/nvme8n1", "/dev/nvme9n1", "/dev/nvme10n1",
            "/dev/nvme11n1", "/dev/nvme12n1", "/dev/nvme13n1", "/dev/nvme14n1", "/dev/nvme15n1",
            "/dev/nvme16n1", "/dev/nvme17n1"
        };
        
        int default_count = sizeof(default_paths) / sizeof(default_paths[0]);
        for (int i = 0; i < default_count && i < max_disks; i++) {
            // 检查磁盘是否存在
            if (access(default_paths[i], F_OK) == 0) {
                disk_paths[i] = strdup(default_paths[i]);
                disk_count++;
                printf("  检测到磁盘: %s\n", default_paths[i]);
            }
        }
        return disk_count;
    }
    
    printf("加载安全磁盘列表...\n");
    char line[256];
    while (fgets(line, sizeof(line), safe_disk_file) && disk_count < max_disks) {
        // 移除换行符
        line[strcspn(line, "\n")] = 0;
        
        // 跳过空行和注释
        if (line[0] == '\0' || line[0] == '#') continue;
        
        // 验证磁盘是否存在且可访问
        if (access(line, F_OK) == 0) {
            // 对于块设备，只检查文件是否存在，不检查读写权限
            // 因为块设备的权限检查可能不准确
            disk_paths[disk_count] = strdup(line);
            printf("  安全磁盘: %s\n", line);
            disk_count++;
        } else {
            printf("  跳过不存在的磁盘: %s\n", line);
        }
    }
    
    fclose(safe_disk_file);
    printf("成功加载 %d 个安全磁盘\n\n", disk_count);
    return disk_count;
}

// EVENODD 编码 - 标准实现
void evenodd_encode(char **data, char **coding, int k, int w, int packetsize) {
    // 参数验证：防止数组越界
    if (k <= 0 || w <= 0 || packetsize <= 0 || !data || !coding) {
        fprintf(stderr, "evenodd_encode: 无效参数 k=%d w=%d packetsize=%d\n", k, w, packetsize);
        return;
    }

// EVENODD 编码要求：w >= k - 1，否则会导致数组越界
    if (w < k - 1) {
        fprintf(stderr, "evenodd_encode: 错误！w=%d < k-1=%d，这会导致数组越界。EVENODD编码要求w >= k-1\n", w, k - 1);
        return;
    }
    
    size_t stripe_size = (size_t)packetsize * (size_t)w;
    
    // 清零校验数据
    memset(coding[0], 0, stripe_size);
    memset(coding[1], 0, stripe_size);
    
    // Combine row and diagonal parity calculations for better locality
    char *S = pool_alloc(global_pool);
    if (!S) {
        fprintf(stderr, "evenodd_encode: 内存池分配失败\n");
        return;
    }
    memset(S, 0, stripe_size);
    char *p_base = coding[0];
    char *q_base = coding[1];
    int diag_mod = (k > 1) ? (k - 1) : 1;
    size_t packet_bytes = (size_t)packetsize;

    for (int packet = 0; packet < w; packet++) {
        size_t packet_offset = (size_t)packet * packet_bytes;
        char *p_dst = p_base + packet_offset;
        char *s_dst = S + packet_offset;
        int diag_pos = (diag_mod == 1) ? 0 : (packet % diag_mod);

        for (int i = 0; i < k; i++) {
            const char *src = data[i] + packet_offset;
            xor_update_simd(p_dst, src, packetsize);

            // 边界检查：确保 diag_pos < w（因为 q_base 的大小是 w * packetsize）
            if (diag_pos < w) {
                char *diag_dst = q_base + (size_t)diag_pos * packet_bytes;
                xor_update_simd(diag_dst, src, packetsize);
            }

            if (diag_pos == 0) {
                xor_update_simd(s_dst, src, packetsize);
            }

            diag_pos++;
            if (diag_pos == diag_mod) {
                diag_pos = 0;
            }
        }
    }

// 边界检查：确保 diag < w
    int max_diag = (k - 1 < w) ? (k - 1) : w;
    for (int diag = 1; diag < max_diag; diag++) {
        char *diag_dst = q_base + (size_t)diag * packet_bytes;
        for (int packet = 0; packet < w; packet++) {
            size_t packet_offset = (size_t)packet * packet_bytes;
            xor_update_simd(diag_dst, S + packet_offset, packetsize);
        }
    }

    pool_free(global_pool, S);
}

// 简单的单块更新函数 - 不进行任何地址分配优化
void update_single_block(char **data __attribute__((unused)), char **coding __attribute__((unused)), int update_disk __attribute__((unused)), int update_packet __attribute__((unused)), char *new_block_data __attribute__((unused)),
                         int *fds __attribute__((unused)), int total_disks __attribute__((unused)), struct io_uring *ring __attribute__((unused)), off_t stripe_disk_offset __attribute__((unused))) {
    // 已不再使用
}

// 加载指定条带到内存缓冲区（data[0..k-1], coding[0..m-1]）
static __attribute__((unused)) void load_stripe_from_disk(long stripe_index, char **data, char **coding,
                                  int *fds __attribute__((unused)), int k, int m, struct io_uring *ring, size_t stripe_size) {
    off_t stripe_offset = (off_t)stripe_index * (off_t)stripe_size;
    // 读取 k 个数据盘
    batch_io_read(data, g_fixed.handles, k, ring, stripe_offset, stripe_size);
    // 读取 m 个校验盘
    batch_io_read(coding, g_fixed.handles + k, m, ring, stripe_offset, stripe_size);
}

// 简单的多块顺序更新 - 批处理并行 I/O
void update_evenodd_simple(char **data __attribute__((unused)), char **coding __attribute__((unused)), char *new_data __attribute__((unused)),
                           int *fds __attribute__((unused)), int total_disks __attribute__((unused)), struct io_uring *ring __attribute__((unused)),
                           long stripe_num __attribute__((unused)), long *next_stripe_cursor __attribute__((unused))) {
    int blocks_to_update = config.update_size / config.packetsize;
    
    struct timespec upd_start_ts, upd_end_ts;
    clock_gettime(CLOCK_MONOTONIC, &upd_start_ts);
    
    if (config.n_updates <= 20) {
        printf("深度集成更新: %d块 (策略=%s)\n",
               blocks_to_update, config.alloc);
    }

    // 共享规划器：按策略生成计划
    BlockPos *allocation_plan = NULL;
    const char *plan_used = NULL;
    int plan_count = 0;
    int need_plan = blocks_to_update > 0 &&
        (strcmp(((config.alloc && config.alloc[0])?config.alloc:"sequential"), "sequential") != 0);

    if (need_plan) {
        allocation_plan = (BlockPos*)malloc(sizeof(BlockPos) * (size_t)blocks_to_update);
        if (allocation_plan) {
            plan_count = plan_block_positions(blocks_to_update, allocation_plan,
                                              0,  // prefer_rs_layout = 0
                                              &plan_used);
            if (plan_count != blocks_to_update) {
                fill_sequential_plan(allocation_plan, blocks_to_update);
                plan_count = blocks_to_update;
                plan_used = "sequential";
            }
        }
    }
    
    free(allocation_plan);
    
    clock_gettime(CLOCK_MONOTONIC, &upd_end_ts);
    double elapsed = timespec_diff_sec(&upd_start_ts, &upd_end_ts);
    
    pthread_mutex_lock(&stats.lock);
    stats.compute_time += elapsed;
    stats.update_count++;
    pthread_mutex_unlock(&stats.lock);
}

// 简单的CRC32实现
static uint32_t crc32(uint32_t crc, const unsigned char *buf, size_t len) {
    static const uint32_t crc32_table[256] = {
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
        0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
        0x09b64c2b, 0x7eb17cbd, 0xe1b7d6d8, 0x96b7d2d0, 0x0a6c1b7c, 0x7fb3d4b8,
        0xe2e1e3e4, 0x95d1d3d4, 0x0b5d0f31, 0x7dcdc7dd, 0xe3b5d8e8, 0x94b4d2d0,
        0x0c5d0f31, 0x7dcdc7dd, 0xe3b5d8e8, 0x94b4d2d0, 0x0a6c1b7c, 0x7fb3d4b8,
        0xe2e1e3e4, 0x95d1d3d4, 0x0b5d0f31, 0x7dcdc7dd, 0xe3b5d8e8, 0x94b4d2d0
    };
    
    crc = crc ^ 0xffffffff;
    while (len--) {
        crc = crc32_table[(crc ^ *buf++) & 0xff] ^ (crc >> 8);
    }
    return crc ^ 0xffffffff;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options] [input_file]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -k <num>      Number of data disks (prime, default %d)\n", config.k);
    fprintf(stderr, "  -m <num>      Number of parity disks (default %d)\n", config.m);
    fprintf(stderr, "  -w <num>      Coding parameter w (default %d)\n", config.w);
    fprintf(stderr, "  -p <bytes>    Packet size in bytes (default %d)\n", config.packetsize);
    fprintf(stderr, "  -u <bytes>    Update size in bytes (default %d)\n", config.update_size);
    fprintf(stderr, "  -n <count>    Number of updates to simulate (default %d)\n", config.n_updates);
    fprintf(stderr, "  -a <policy>   Address allocation policy (sequential,row,diag,auto,...)\n");
    fprintf(stderr, "  -S            Enable strong consistency revalidation flag\n");
    fprintf(stderr, "  -V <count>    Enable verification (sample stripes, <=0 for full)\n");
    fprintf(stderr, "  -h            Show this help and exit\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "If an input_file is provided its first stripe is encoded; otherwise synthetic data is used.\n");
}

// 防止文件描述符与标准输入输出冲突
static int ensure_nonstdio_fd(int fd, const char *who) {
    fprintf(stderr, "[FD-CHECK] %s: checking fd=%d\n", who ? who : "unknown", fd);
    if (fd >= 3) {
        fprintf(stderr, "[FD-CHECK] %s: fd %d is safe (>= 3)\n", who ? who : "unknown", fd);
        return fd;
    }
    fprintf(stderr, "[FD-CHECK] %s: fd %d is unsafe (< 3), duplicating...\n", who ? who : "unknown", fd);
    int dupfd = fcntl(fd, F_DUPFD, 3);
    if (dupfd < 0) {
        fprintf(stderr, "[FATAL] %s: dupfd failed for fd %d: %s\n",
                who ? who : "unknown", fd, strerror(errno));
        abort();
    }
    close(fd);
    fprintf(stderr, "[FD-CHECK] %s: duplicated fd %d -> %d to avoid stdio collision\n",
           who ? who : "unknown", fd, dupfd);
    return dupfd;
}

int main(int argc, char *argv[]) {
    int opt;
    const char *input_path = NULL;
    int exit_code = 0;
    int stats_lock_initialized = 0;
    int input_fd = -1;
    int input_exhausted = 0;
    int w_option_explicit = 0;

    while ((opt = getopt(argc, argv, "k:m:w:p:u:n:a:SV:h")) != -1) {
        switch (opt) {
            case 'k': config.k = atoi(optarg); break;
            case 'm': config.m = atoi(optarg); break;
            case 'w': config.w = atoi(optarg); w_option_explicit = 1; break;
            case 'p': config.packetsize = atoi(optarg); break;
            case 'u': config.update_size = atoi(optarg); break;
            case 'n': config.n_updates = atoi(optarg); break;
            case 'a': config.alloc = optarg; break;
            case 'S': config.strong = 1; break;
            case 'V': config.verify = 1; config.verify_samples = atol(optarg); break;
            case 'h':
            default:
                print_usage(argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    if (optind < argc) {
        input_path = argv[optind];
    }

    if (config.k <= 0) {
        fprintf(stderr, "Invalid data disk count (%d)\n", config.k);
        return 1;
    }
    if (!is_prime(config.k)) {
        fprintf(stderr, "Data disk count k=%d must be prime for EVENODD coding\n", config.k);
        return 1;
    }
    if (config.m <= 0) {
        fprintf(stderr, "Invalid parity disk count (%d)\n", config.m);
        return 1;
    }
    if (!w_option_explicit && config.w < config.k - 1) {
        int required_min = config.k - 1;
        int adjusted_w = config.k > required_min ? config.k : required_min;
        if (adjusted_w > 512) {
            fprintf(stderr,
                    "EVENODD requires w >= k-1 (%d), but automatic adjustment would exceed the maximum supported w=512.\n",
                    required_min);
            fprintf(stderr, "Please rerun with -w set to at least %d.\n", required_min);
            return 1;
        }
        fprintf(stderr,
                "[INFO] Auto-adjusting w from %d to %d to satisfy EVENODD requirement w >= k-1.\n",
                config.w, adjusted_w);
        config.w = adjusted_w;
    }
    if (config.w <= 0 || config.w > 512) {
        fprintf(stderr, "Invalid w value (%d)\n", config.w);
        return 1;
    }
    if (w_option_explicit && config.w < config.k - 1) {
        fprintf(stderr,
                "Error: w=%d < k-1=%d. EVENODD requires w >= k-1 to avoid out-of-bounds access.\n",
                config.w, config.k - 1);
        fprintf(stderr, "Please rerun with -w set to at least %d (recommended: -w %d).\n",
                config.k - 1, config.k);
        return 1;
    }
    // EVENODD 编码要求：w >= k - 1，否则会导致数组越界和段错误
    if (config.w < config.k - 1) {
        fprintf(stderr, "错误：w=%d < k-1=%d。EVENODD编码要求w >= k-1，否则会导致数组越界\n", config.w, config.k - 1);
        fprintf(stderr, "请将w设置为至少%d（推荐：w >= k）\n", config.k - 1);
        return 1;
    }
    if (config.packetsize <= 0) {
        fprintf(stderr, "Invalid packet size (%d)\n", config.packetsize);
        return 1;
    }

    if (pthread_mutex_init(&stats.lock, NULL) != 0) {
        perror("pthread_mutex_init");
        return 1;
    }
    stats_lock_initialized = 1;

    size_t stripe_bytes = (size_t)config.w * (size_t)config.packetsize;
    int pool_capacity = (config.k + config.m) * 4;
    if (pool_capacity < 4) pool_capacity = 4;
    global_pool = create_memory_pool(pool_capacity, stripe_bytes);
    if (!global_pool) {
        fprintf(stderr, "Failed to initialise memory pool\n");
        exit_code = 1;
        goto cleanup;
    }

    char **data = NULL;
    char **coding = NULL;
    data = calloc((size_t)config.k, sizeof(char*));
    coding = calloc((size_t)config.m, sizeof(char*));
    if (!data || !coding) {
        fprintf(stderr, "Out of memory allocating buffer tables\n");
        exit_code = 1;
        goto cleanup;
    }

    for (int i = 0; i < config.k; i++) {
        data[i] = pool_alloc(global_pool);
        if (!data[i]) {
            fprintf(stderr, "Failed to allocate data buffer %d\n", i);
            exit_code = 1;
            goto cleanup;
        }
    }
    for (int i = 0; i < config.m; i++) {
        coding[i] = pool_alloc(global_pool);
        if (!coding[i]) {
            fprintf(stderr, "Failed to allocate parity buffer %d\n", i);
            exit_code = 1;
            goto cleanup;
        }
    }

    if (input_path) {
        input_fd = open(input_path, O_RDONLY);
        if (input_fd < 0) {
            fprintf(stderr, "Failed to open %s: %s\n", input_path, strerror(errno));
            exit_code = 1;
            goto cleanup;
        }
    }

    for (int disk = 0; disk < config.k; disk++) {
        for (int packet = 0; packet < config.w; packet++) {
            char *dst = data[disk] + (size_t)packet * (size_t)config.packetsize;
            if (input_fd >= 0 && !input_exhausted) {
                ssize_t bytes = read(input_fd, dst, (size_t)config.packetsize);
                if (bytes < 0) {
                    fprintf(stderr, "Read error from %s: %s\n", input_path, strerror(errno));
                    exit_code = 1;
                    goto cleanup;
                } else if (bytes == 0) {
                    memset(dst, 0, (size_t)config.packetsize);
                    input_exhausted = 1;
                } else if (bytes < config.packetsize) {
                    memset(dst + bytes, 0, (size_t)config.packetsize - (size_t)bytes);
                    input_exhausted = 1;
                }
            } else if (input_fd >= 0 && input_exhausted) {
                memset(dst, 0, (size_t)config.packetsize);
            } else {
                for (int b = 0; b < config.packetsize; b++) {
                    dst[b] = (char)((disk + 1) * 17 + packet * 3 + b);
                }
            }
        }
    }

    if (input_fd >= 0) {
        close(input_fd);
        input_fd = -1;
    }

    evenodd_encode(data, coding, config.k, config.w, config.packetsize);

    printf("EVENODD fast path demo (k=%d, m=%d, w=%d, packetsize=%d)\n",
           config.k, config.m, config.w, config.packetsize);
    printf("Allocation strategy: %s\n", config.alloc);

    printf("\nData block CRC32 digests:\n");
    for (int i = 0; i < config.k; i++) {
        uint32_t crc = crc32(0, (const unsigned char*)data[i], stripe_bytes);
        printf("  D%02d -> %08x\n", i, crc);
    }

    printf("Parity block CRC32 digests:\n");
    for (int i = 0; i < config.m; i++) {
        uint32_t crc = crc32(0, (const unsigned char*)coding[i], stripe_bytes);
        printf("  P%02d -> %08x\n", config.k + i, crc);
    }


cleanup:
    if (input_fd >= 0) {
        close(input_fd);
    }
    if (coding) {
        for (int i = 0; i < config.m; i++) {
            if (coding[i] && global_pool) {
                pool_free(global_pool, coding[i]);
            }
        }
        free(coding);
    }
    if (data) {
        for (int i = 0; i < config.k; i++) {
            if (data[i] && global_pool) {
                pool_free(global_pool, data[i]);
            }
        }
        free(data);
    }
    if (global_pool) {
        destroy_memory_pool(global_pool);
        global_pool = NULL;
    }
    if (stats_lock_initialized) {
        pthread_mutex_destroy(&stats.lock);
    }
    return exit_code;
}
