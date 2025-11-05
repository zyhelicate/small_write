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

#ifndef PARIX_DEBUG_LOG
#define PARIX_DEBUG_LOG 0
#endif

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
static int rs_mmap_init(void);
static void rs_mmap_destroy(void);
static uint32_t crc32(uint32_t crc, const unsigned char *buf, size_t len);
static off_t rs_get_stripe_base_offset_optimized(int stripe_id, int parity_type);

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
    int use_rs;         // 是否使用Reserved Space进行校验日志化
    int use_parix;      // 是否启用PARIX推测性部分写入
    int parix_threshold;// PARIX推测阈值（连续写入次数）
} config_t;

// Reserved Space (RS) 相关结构
#define RS_DEFAULT_SIZE (4 * 1024 * 1024)  // 每条带RS默认大小: 4MB（与设备侧预留区一致）
#define RS_LOG_ENTRY_MAX_SIZE 4096    // 单个日志条目最大大小

// 日志条目结构
typedef struct {
    uint32_t magic;           // 魔数标识: 0x5253504C ('RSPL')
    uint32_t stripe_id;       // 条带ID
    uint16_t disk_id;         // 磁盘ID (0..k-1 为数据盘, k为P盘, k+1为Q盘)
    uint16_t packet_id;       // 包ID (0..w-1)
    uint32_t delta_size;      // delta数据大小
    uint32_t sequence;        // 序列号
    uint32_t checksum;        // 校验和
    uint8_t parity_type;      // 校验类型: 0=P, 1=Q
    uint8_t reserved[3];      // 对齐保留
    // 后续跟随delta数据
} __attribute__((packed)) rs_log_entry_t;

// PARIX推测性日志条目结构
typedef struct {
    uint32_t magic;           // 魔数标识: 0x50415249 ('PARI')
    uint32_t stripe_id;       // 条带ID
    uint16_t disk_id;         // 磁盘ID (0..k-1 为数据盘)
    uint16_t packet_id;       // 包ID (0..w-1)
    uint32_t data_size;       // 新数据大小
    uint32_t sequence;        // 序列号
    uint32_t checksum;        // 校验和
    uint8_t is_initial;       // 是否为初始写入（需要读取旧值）
    uint8_t reserved[3];      // 对齐保留
    // 后续跟随新数据
} __attribute__((packed)) parix_log_entry_t;

// 条带RS元数据
typedef struct {
    uint64_t p_rs_offset;     // P盘RS当前偏移（64位）
    uint64_t q_rs_offset;     // Q盘RS当前偏移（64位）
    uint64_t p_rs_size;       // P盘RS总大小（64位）
    uint64_t q_rs_size;       // Q盘RS总大小（64位）
    uint64_t next_sequence;   // 下一个序列号（64位）
    uint64_t log_count;       // 当前日志条目数（64位）
    uint64_t packet_bitmap[8]; // 条带内被触及的packet位图（最多支持w<=512）
    pthread_mutex_t p_lock;   // P 偏移锁
    pthread_mutex_t q_lock;   // Q 偏移锁
} stripe_rs_meta_t;

// PARIX推测性元数据
typedef struct {
    // 数据块推测状态：记录每个数据块的推测信息
    struct {
        int is_speculative;   // 是否处于推测状态
        uint32_t base_sequence; // 基础序列号（首次写入时）
        char *base_data;      // 基础数据（首次写入时的数据）
        uint32_t write_count; // 连续写入次数
        struct timespec last_write; // 最后写入时间
        int allocation_hint; // 地址分配提示 (0=row, 1=diag, 2=mixed)
        int speculation_priority; // 推测优先级 (0-10, 10最高)
    } data_blocks[16][16];    // 支持最大k=16, w=16
    
    // 推测性日志偏移
    uint32_t parix_offset;    // PARIX日志当前偏移
    uint32_t parix_size;      // PARIX日志总大小
    uint32_t parix_sequence;  // PARIX序列号
    uint32_t parix_count;     // PARIX日志条目数
    
    // 地址分配协同信息
    struct {
        int current_allocation_strategy; // 当前使用的分配策略
        int row_blocks_count;           // 行分配块数量
        int diag_blocks_count;          // 对角线分配块数量
        int mixed_blocks_count;         // 混合分配块数量
        double allocation_efficiency;   // 分配效率 (0.0-1.0)
        int hotspot_rows[16];           // 热点行标识
        int hotspot_diags[16];          // 热点对角线标识
    } allocation_meta;
    
    pthread_mutex_t parix_lock; // PARIX锁
} stripe_parix_meta_t;

// 全局RS管理
typedef struct {
    stripe_rs_meta_t *stripe_metas;  // 每条带的RS元数据
    int total_stripes;               // 总条带数
    off_t rs_base_offset;            // RS在设备上的基础偏移
    int rs_enabled;                  // RS功能是否启用
    int merge_threshold;             // 合并阈值(百分比)
    pthread_t merge_thread;          // 后台合并线程
    int merge_thread_running;        // 合并线程运行标志
    int rs_fd;                       // RS专用文件描述符
    char rs_file_path[256];          // RS文件路径
} rs_manager_t;

// 全局PARIX管理
typedef struct {
    stripe_parix_meta_t *stripe_metas; // 每条带的PARIX元数据
    int total_stripes;                 // 总条带数
    off_t parix_base_offset;          // PARIX在设备上的基础偏移
    int parix_enabled;                // PARIX功能是否启用
    int parix_fd;                     // PARIX专用文件描述符
    char parix_file_path[256];        // PARIX文件路径
    pthread_t replay_thread;          // 后台重放线程
    int replay_thread_running;        // 重放线程运行标志
    pthread_mutex_t replay_mu;        // 条件变量互斥
    pthread_cond_t replay_cv;         // 条件变量
    int need_replay;                  // 触发标志
} parix_manager_t;

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

// 优化版 RS 接口（后台聚合刷盘）。当启用 -R 时使用
// 说明：这里直接在本文件内提供实现，封装到本地RS管理器（g_rs_manager）。
static void rs_init(int threads);
static void rs_destroy(void);
static __attribute__((unused)) int rs_log_write(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id,
                        uint8_t parity_type, void *delta, uint32_t delta_size);

// PARIX推测性部分写入接口
static void parix_init(int threads);
static void parix_destroy(void);
static int parix_speculative_write(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id,
                                   void *new_data, uint32_t data_size);
static int parix_replay_stripe(uint32_t stripe_id);
static int parix_should_speculate(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id);
static void* parix_replay_thread_func(void *arg);
static int parix_speculative_write_enhanced(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id,
                                            void *new_data, uint32_t data_size, BlockPos *allocation_context);

// PARIX与地址分配策略协同接口
static int parix_analyze_allocation_pattern(uint32_t stripe_id, BlockPos *blocks, int block_count);
static int parix_smart_speculation(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id, 
                                   BlockPos *allocation_context);
static int parix_adaptive_threshold(uint32_t stripe_id, int allocation_strategy);
static void parix_update_allocation_meta(uint32_t stripe_id, BlockPos *blocks, int block_count);
static int parix_predict_speculation_benefit(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id);
static int parix_manager_init_optimized(int total_stripes, off_t base_offset);
static void parix_manager_destroy_optimized(void);


static int rs_write_log_entries_batch(
    int stripe_id,
    int *disk_ids,
    int *packet_ids,
    uint8_t *parity_types,
    void **delta_datas,
    uint32_t *delta_sizes,
    int entry_count,
    struct io_uring *ring,
    int *fds
);

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
    .strong = 0,
    .use_rs = 0
};

perf_stats_t stats = {0};
memory_pool_t *global_pool = NULL;

// RS管理器全局实例
static rs_manager_t g_rs_manager = {0};

// PARIX管理器全局实例
static parix_manager_t g_parix_manager = {0};

// PARIX前台写入的 io_uring 环境（可选）
static struct io_uring g_parix_ring;
static int g_parix_ring_ready = 0;

// 设备侧预留区布局（默认值，按用户确认）
static const off_t P_RS_BASE_OFFSET = (off_t)(1ULL << 30);        // 1 GiB
static const off_t Q_RS_BASE_OFFSET = (off_t)(2ULL << 30);        // 2 GiB
static const off_t RS_AREA_PER_STRIPE = (off_t)(64ULL * 1024ULL * 1024ULL); // 64 MiB（扩大，减少wrap）
static const off_t PARIX_BASE_OFFSET = (off_t)(3ULL << 30);       // 3 GiB (落在P盘)
static const off_t PARIX_AREA_PER_STRIPE = (off_t)(4ULL * 1024ULL * 1024ULL); // 4 MiB

// RS内存映射全局变量
static void *g_rs_mmap = NULL;
static size_t g_rs_mmap_size = 0;

// ========== RS聚合器（将多次小delta聚合为条带批量顺序写） ==========
typedef struct {
    uint32_t stripe_id;
    int count;
    int capacity;
    uint16_t *disk_ids;
    uint16_t *packet_ids;
    uint8_t  *types;
    void    **datas;
    uint32_t *sizes;
    uint8_t  *from_pool;
    // 扩展：基于时间窗口的聚合
    struct timespec window_start;
    int max_batch;           // 触发冲刷的最大条目数阈值
    int sort_enabled;        // 冲刷前是否按 (disk,packet,type) 排序以提升顺序性
    int window_ms;           // 时间窗口（毫秒），超过则冲刷
} rs_batch_t;

// 优化：多条带并行RS聚合器（提升RS吞吐量）
typedef struct {
    rs_batch_t *stripe_batches;  // 每个条带一个批次
    int max_stripes;             // 最大并发条带数
    int active_count;            // 当前活跃条带数
    pthread_mutex_t lock;        // 保护并发访问
    struct io_uring *shared_ring; // 共享io_uring实例
    int total_entries;           // 所有条带的总条目数
    int global_max_batch;        // 全局批量阈值（跨条带）
} rs_multi_batch_t;

static rs_batch_t g_rs_batch = {0};
static rs_multi_batch_t g_rs_multi = {0};
static pthread_mutex_t g_rs_batch_lock = PTHREAD_MUTEX_INITIALIZER;
static memory_pool_t *rs_delta_pool = NULL;

// RS 专用 io_uring（用于批量冲刷），按是否就绪选择性传入
static struct io_uring g_rs_ring;
static int g_rs_ring_ready = 0;

// 分片批次结构（减少全局锁争用）
typedef struct {
    rs_batch_t batch;
    pthread_mutex_t lock;
} rs_shard_t;

static rs_shard_t *g_rs_shards = NULL;
static int g_rs_shard_count = 0; // 必须为2的幂以便位运算取模

static inline int rs_pick_shard(uint32_t stripe_id) {
    if (g_rs_shard_count <= 0) return -1;
    return (int)(stripe_id & (uint32_t)(g_rs_shard_count - 1));
}

// 判断是否应当触发冲刷（不产生副作用）
static int rs_batch_should_flush(rs_batch_t *b) {
    if (!b) return 0;
    if (b->count >= b->max_batch) return 1;
    struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
    long ms = (now.tv_sec - b->window_start.tv_sec) * 1000 + (now.tv_nsec - b->window_start.tv_nsec) / 1000000;
    // 小批次（<256）不因时间窗口触发刷新，避免频繁小批量
    if (b->count >= 256 && ms >= b->window_ms) return 1;
    return 0;
}

// 将源批次的数据结构移动到目标批次，源批次被快速重置以便继续接收
static void rs_batch_detach(rs_batch_t *src, rs_batch_t *dst) {
    memset(dst, 0, sizeof(*dst));
    // 直接转移指针所有权
    dst->stripe_id  = src->stripe_id;
    dst->count      = src->count;
    dst->capacity   = src->capacity;
    dst->disk_ids   = src->disk_ids;   src->disk_ids = NULL;
    dst->packet_ids = src->packet_ids; src->packet_ids = NULL;
    dst->types      = src->types;      src->types = NULL;
    dst->datas      = src->datas;      src->datas = NULL;
    dst->sizes      = src->sizes;      src->sizes = NULL;
    dst->from_pool  = src->from_pool;  src->from_pool = NULL;
    dst->max_batch  = src->max_batch;
    dst->sort_enabled = src->sort_enabled;
    dst->window_ms  = src->window_ms;
    // 快速重置源批次（保留配置，但清空内容）
    src->stripe_id = (uint32_t)~0u;
    src->count = 0;
    clock_gettime(CLOCK_MONOTONIC, &src->window_start);
    src->capacity = 0;
}

// 释放批次数组（条目数据缓冲已在 flush 中释放，这里仅释放指针数组）
static void rs_batch_free_arrays(rs_batch_t *b) {
    if (!b) return;
    if (b->disk_ids)  { free(b->disk_ids);  b->disk_ids = NULL; }
    if (b->packet_ids){ free(b->packet_ids);b->packet_ids = NULL; }
    if (b->types)     { free(b->types);     b->types = NULL; }
    if (b->datas)     { free(b->datas);     b->datas = NULL; }
    if (b->sizes)     { free(b->sizes);     b->sizes = NULL; }
    if (b->from_pool) { free(b->from_pool); b->from_pool = NULL; }
    b->capacity = 0;
    b->count = 0;
}

static void rs_batch_reset(rs_batch_t *b) {
    b->stripe_id = (uint32_t)~0u;
    b->count = 0;
    clock_gettime(CLOCK_MONOTONIC, &b->window_start);
    // 提升吞吐：扩大批量阈值，延长窗口（小批次不靠时间刷新）
    if (b->max_batch <= 0) b->max_batch = 4096;   // 默认 4096 条
    if (b->window_ms <= 0) b->window_ms = 8;      // 默认 8ms 窗口
}

// 优化：初始化多条带RS聚合器
static int rs_multi_batch_init(rs_multi_batch_t *mb, int max_stripes, struct io_uring *ring) {
    memset(mb, 0, sizeof(*mb));
    mb->max_stripes = max_stripes;
    mb->global_max_batch = 2048;  // 提升全局批量阈值
    mb->shared_ring = ring;
    mb->stripe_batches = (rs_batch_t*)calloc((size_t)max_stripes, sizeof(rs_batch_t));
    if (!mb->stripe_batches) return -1;
    for (int i = 0; i < max_stripes; i++) {
        rs_batch_reset(&mb->stripe_batches[i]);
        mb->stripe_batches[i].max_batch = 1024;  // 单条带批量
        mb->stripe_batches[i].window_ms = 2;     // 更短窗口
        mb->stripe_batches[i].sort_enabled = 1;
    }
    pthread_mutex_init(&mb->lock, NULL);
    return 0;
}

// 优化：销毁多条带RS聚合器
static void rs_multi_batch_destroy(rs_multi_batch_t *mb) {
    if (mb->stripe_batches) {
        for (int i = 0; i < mb->max_stripes; i++) {
            rs_batch_t *b = &mb->stripe_batches[i];
            if (b->disk_ids) free(b->disk_ids);
            if (b->packet_ids) free(b->packet_ids);
            if (b->types) free(b->types);
            if (b->datas) {
                for (int j = 0; j < b->count; j++) {
                    if (!b->datas[j]) continue;
                    if (b->from_pool && b->from_pool[j]) {
                        pool_free(rs_delta_pool, b->datas[j]);
                    } else {
                        free(b->datas[j]);
                    }
                }
                free(b->datas);
            }
            if (b->sizes) free(b->sizes);
            if (b->from_pool) free(b->from_pool);
        }
        free(mb->stripe_batches);
    }
    pthread_mutex_destroy(&mb->lock);
    memset(mb, 0, sizeof(*mb));
}

static int rs_batch_ensure(rs_batch_t *b, int need) {
    if (b->capacity >= need) return 0;
    int nc = b->capacity ? b->capacity * 2 : 256;
    while (nc < need) nc *= 2;
    void *tmp;
    tmp = realloc(b->disk_ids, sizeof(uint16_t) * (size_t)nc);
    if (!tmp) return -1;
    b->disk_ids = tmp;
    tmp = realloc(b->packet_ids, sizeof(uint16_t) * (size_t)nc);
    if (!tmp) return -1;
    b->packet_ids = tmp;
    tmp = realloc(b->types, sizeof(uint8_t) * (size_t)nc);
    if (!tmp) return -1;
    b->types = tmp;
    tmp = realloc(b->datas, sizeof(void*) * (size_t)nc);
    if (!tmp) return -1;
    b->datas = tmp;
    tmp = realloc(b->sizes, sizeof(uint32_t) * (size_t)nc);
    if (!tmp) return -1;
    b->sizes = tmp;
    tmp = realloc(b->from_pool, sizeof(uint8_t) * (size_t)nc);
    if (!tmp) return -1;
    b->from_pool = tmp;
    memset(b->from_pool + b->capacity, 0, (size_t)(nc - b->capacity));
    b->capacity = nc;
    return 0;
}



static int rs_batch_flush(rs_batch_t *b, struct io_uring *ring) {
    if (b->count == 0 || b->stripe_id == (uint32_t)~0u) return 0;
    // 小批次快速路径：直接写出，跳过排序与就地合并，降低CPU开销
    if (b->count < 64) {
        int ret = rs_write_log_entries_batch((int)b->stripe_id,
                                             (int*)b->disk_ids,
                                             (int*)b->packet_ids,
                                             b->types,
                                             b->datas,
                                             b->sizes,
                                             b->count,
                                             ring,
                                             NULL);
        for (int i = 0; i < b->count; i++) {
            if (b->datas && b->datas[i]) {
                if (b->from_pool && b->from_pool[i]) pool_free(rs_delta_pool, b->datas[i]);
                else free(b->datas[i]);
                b->datas[i] = NULL;
                if (b->from_pool) b->from_pool[i] = 0;
            }
        }
        rs_batch_reset(b);
        return ret < 0 ? ret : 0;
    }
    // 可选：按 (disk_id, packet_id, type) 排序，提升顺序写局部性（使用 qsort 提升性能）
    if (b->sort_enabled && b->count > 1) {
        typedef struct { uint16_t d, p; uint8_t t; void *data; uint32_t sz; } sort_item_t;
        sort_item_t *arr = (sort_item_t*)alloca(sizeof(sort_item_t) * (size_t)b->count);
        for (int i = 0; i < b->count; i++) {
            arr[i].d = b->disk_ids[i];
            arr[i].p = b->packet_ids[i];
            arr[i].t = b->types[i];
            arr[i].data = b->datas[i];
            arr[i].sz = b->sizes[i];
        }
        int cmp(const void *a, const void *b_) {
            const sort_item_t *x = (const sort_item_t*)a; const sort_item_t *y = (const sort_item_t*)b_;
            if (x->d != y->d) return (int)x->d - (int)y->d;
            if (x->p != y->p) return (int)x->p - (int)y->p;
            return (int)x->t - (int)y->t;
        }
        qsort(arr, (size_t)b->count, sizeof(sort_item_t), cmp);
        for (int i = 0; i < b->count; i++) {
            b->disk_ids[i] = arr[i].d;
            b->packet_ids[i] = arr[i].p;
            b->types[i] = arr[i].t;
            b->datas[i] = arr[i].data;
            b->sizes[i] = arr[i].sz;
        }
    }
    // 新增：在排序后做就地聚合，将相同 (disk, packet, type) 的多条 delta XOR 合并，减少RS写入次数
    if (b->count > 1) {
        int w = 0; // 写指针
        for (int r = 0; r < b->count; ) {
            int nr = r + 1;
            // 找到相同 key 的区间 [r, nr)
            while (nr < b->count &&
                   b->disk_ids[nr] == b->disk_ids[r] &&
                   b->packet_ids[nr] == b->packet_ids[r] &&
                   b->types[nr] == b->types[r]) {
                nr++;
            }
            // 优化：使用SIMD加速XOR合并
            for (int j = r + 1; j < nr; j++) {
                unsigned char *dst = (unsigned char*)b->datas[r];
                unsigned char *src = (unsigned char*)b->datas[j];
                uint32_t sz = b->sizes[r];
                if (b->sizes[j] < sz) sz = b->sizes[j];
                xor_update_simd((char*)dst, (char*)src, sz);  // 使用SIMD
                if (b->from_pool && b->from_pool[j]) {
                    pool_free(rs_delta_pool, b->datas[j]);
                } else {
                    free(b->datas[j]);
                }
                b->datas[j] = NULL;
                if (b->from_pool) b->from_pool[j] = 0;
            }
            // 将合并后的 r 元素挪到位置 w
            if (w != r) {
                b->disk_ids[w] = b->disk_ids[r];
                b->packet_ids[w] = b->packet_ids[r];
                b->types[w] = b->types[r];
                b->datas[w] = b->datas[r]; b->datas[r] = NULL;
                b->sizes[w] = b->sizes[r];
                if (b->from_pool) {
                    b->from_pool[w] = b->from_pool[r];
                    b->from_pool[r] = 0;
                }
            }
            w++;
            r = nr;
        }
        b->count = w;
    }
    int ret = rs_write_log_entries_batch((int)b->stripe_id,
                                         (int*)b->disk_ids,
                                         (int*)b->packet_ids,
                                         b->types,
                                         b->datas,
                                         b->sizes,
                                         b->count,
                                         ring,
                                         NULL);
    // 释放为 delta 复制分配的缓冲
    for (int i = 0; i < b->count; i++) {
        if (b->datas && b->datas[i]) {
            if (b->from_pool && b->from_pool[i]) {
                pool_free(rs_delta_pool, b->datas[i]);
            } else {
                free(b->datas[i]);
            }
            b->datas[i] = NULL;
            if (b->from_pool) b->from_pool[i] = 0;
        }
    }
    rs_batch_reset(b);
    return ret < 0 ? ret : 0;
}

// 优化：跨条带批量冲刷（一次性写入多个条带的RS日志）
static int rs_multi_batch_flush_all(rs_multi_batch_t *mb) {
    if (!mb || !mb->stripe_batches) return 0;
    
    pthread_mutex_lock(&mb->lock);
    
    // 统计需要冲刷的条带数
    int active_stripes = 0;
    for (int i = 0; i < mb->max_stripes; i++) {
        if (mb->stripe_batches[i].count > 0) active_stripes++;
    }
    
    if (active_stripes == 0) {
        pthread_mutex_unlock(&mb->lock);
        return 0;
    }

// 逐个冲刷但使用共享ring提升并发
    int flushed = 0;
    for (int i = 0; i < mb->max_stripes; i++) {
        if (mb->stripe_batches[i].count > 0) {
            int ret = rs_batch_flush(&mb->stripe_batches[i], mb->shared_ring);
            if (ret == 0) flushed++;
        }
    }
    
    mb->total_entries = 0;
    pthread_mutex_unlock(&mb->lock);
    return flushed;
}

static void rs_flush_pending(void) {
    pthread_mutex_lock(&g_rs_batch_lock);
    if (g_rs_batch.count > 0) {
        rs_batch_flush(&g_rs_batch, g_rs_ring_ready ? &g_rs_ring : NULL);
    }
    pthread_mutex_unlock(&g_rs_batch_lock);
}

static int rs_batch_append(rs_batch_t *b, uint32_t stripe_id, uint16_t disk_id,
                           uint16_t packet_id, uint8_t parity_type,
                           void *delta, uint32_t delta_size,
                           struct io_uring *ring) {
    if (b->stripe_id == (uint32_t)~0u) b->stripe_id = stripe_id;
    if (b->stripe_id != stripe_id) {
        // 先冲刷原条带
        int r = rs_batch_flush(b, ring);
        if (r != 0) return r;
        b->stripe_id = stripe_id;
    }
    if (rs_batch_ensure(b, b->count + 1) != 0) return -1;
    b->disk_ids[b->count]  = disk_id;
    b->packet_ids[b->count]= packet_id;
    b->types[b->count]     = parity_type;
    // 复制 delta 到独立缓冲（无需对齐，RS写时会另行对齐封装）
    void *delta_copy = NULL;
    int used_pool = 0;
    if (rs_delta_pool) {
        delta_copy = pool_alloc(rs_delta_pool);
        if (delta_copy) {
            used_pool = 1;
        }
    }
    if (!delta_copy) {
        delta_copy = malloc((size_t)delta_size);
        if (!delta_copy) return -1;
    }
    memcpy(delta_copy, delta, (size_t)delta_size);
    b->datas[b->count]     = delta_copy;
    b->sizes[b->count]     = delta_size;
    if (b->from_pool) b->from_pool[b->count] = (uint8_t)used_pool;
    b->count++;
    
    // 冲刷判定放到上层（锁外）执行
    return 0;
}

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

// ========== Reserved Space (RS) 功能实现 (优化版本) ==========

// 前向声明
static __attribute__((unused)) int rs_should_merge_stripe(int stripe_id);
static int rs_merge_stripe_simple(int stripe_id);
static int rs_merge_stripe_replay(int stripe_id);

// 后台合并线程函数（超级优化版：智能调整检查频率）
static void* rs_merge_thread_func(void* arg __attribute__((unused))) {
    // 根据地址策略调整检查频率
    int check_interval = 30000;  // 30ms检查
    int merge_batch_size = 12;    // 每轮最多合并12个条带
    
    // 优化：根据k值调整后台合并频率（解决k=7,11,13性能问题）
    if (strcmp(config.alloc, "batch_optimized") == 0) {
        check_interval = 15000;   // 15ms
        merge_batch_size = 20;    // 20条带/轮
        printf("RS后台合并: 分配=batch_optimized（%dms，%d条带/轮）\n",
               check_interval/1000, merge_batch_size);
    } else if (strcmp(config.alloc, "row") == 0) {
        check_interval = 25000;   // 25ms
        merge_batch_size = 15;
        printf("RS后台合并: 分配=row（%dms，%d条带/轮）\n",
               check_interval/1000, merge_batch_size);
    } else if (strcmp(config.alloc, "diag") == 0) {
        check_interval = 25000;   // 25ms
        merge_batch_size = 15;
        printf("RS后台合并: 分配=diag（%dms，%d条带/轮）\n",
               check_interval/1000, merge_batch_size);
    } else if (strcmp(config.alloc, "sequential") == 0) {
        check_interval = 30000;   // 30ms
        merge_batch_size = 12;
        printf("RS后台合并: 分配=sequential（%dms，%d条带/轮）\n",
               check_interval/1000, merge_batch_size);
    }
    
    while (g_rs_manager.merge_thread_running) {
        int merged_count = 0;
        
        // 优化：优先合并使用率最高的条带
        typedef struct { int stripe_id; int usage; } stripe_usage_t;
        stripe_usage_t *usages = alloca(sizeof(stripe_usage_t) * g_rs_manager.total_stripes);
        
        for (int i = 0; i < g_rs_manager.total_stripes; i++) {
            stripe_rs_meta_t *meta = &g_rs_manager.stripe_metas[i];
            int p_usage = (meta->p_rs_offset * 100) / meta->p_rs_size;
            int q_usage = (meta->q_rs_offset * 100) / meta->q_rs_size;
            usages[i].stripe_id = i;
            usages[i].usage = (p_usage > q_usage) ? p_usage : q_usage;
        }

// 简单冒泡排序（取前merge_batch_size个）
        for (int i = 0; i < g_rs_manager.total_stripes && i < merge_batch_size; i++) {
            for (int j = i + 1; j < g_rs_manager.total_stripes; j++) {
                if (usages[j].usage > usages[i].usage) {
                    stripe_usage_t tmp = usages[i];
                    usages[i] = usages[j];
                    usages[j] = tmp;
                }
            }
        }

// 合并使用率最高的条带
        for (int i = 0; i < merge_batch_size && i < g_rs_manager.total_stripes && 
             g_rs_manager.merge_thread_running; i++) {
            int stripe_id = usages[i].stripe_id;
            if (usages[i].usage >= g_rs_manager.merge_threshold) {
                rs_merge_stripe_simple(stripe_id);
                merged_count++;
            }
        }

// 休眠一段时间再检查
        usleep(check_interval);
    }
    return NULL;
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

// 判断是否具备真实设备固定句柄以进行磁盘I/O
static inline int rs_fixed_io_ready(void) {
    return (g_fixed.enabled && g_fixed.handles != NULL);
}

// 初始化RS管理器（优化版本，包含后台合并和真实磁盘）
static int rs_manager_init_optimized(int total_stripes, off_t base_offset) {
    g_rs_manager.total_stripes = total_stripes;
    g_rs_manager.rs_base_offset = base_offset;
    g_rs_manager.rs_enabled = 1;
    // 优化：根据k值自适应调整合并阈值（解决k=7,11,13性能问题）
    if (config.k == 5) {
        g_rs_manager.merge_threshold = 85;  // k=5: 85%阈值
    } else if (config.k == 7) {
        g_rs_manager.merge_threshold = 85;  // k=7: 85%阈值（与k=5相同）
    } else if (config.k == 11) {
        g_rs_manager.merge_threshold = 83;  // k=11: 83%阈值（接近k=5）
    } else if (config.k == 13) {
        g_rs_manager.merge_threshold = 80;  // k=13: 80%阈值（提升）
    } else {
        // 其他k值：基于k值动态计算
        g_rs_manager.merge_threshold = 85 - (config.k - 5) * 3;  // 每增加k减少3%
        if (g_rs_manager.merge_threshold < 60) g_rs_manager.merge_threshold = 60;
    }

    // 覆盖：按地址策略设定阈值（与后台线程的检查/批量配合）
    if (strcmp(config.alloc, "batch_optimized") == 0) {
        g_rs_manager.merge_threshold = 90;   // 超激进合并
    } else if (strcmp(config.alloc, "row") == 0) {
        g_rs_manager.merge_threshold = 82;   // 行批量
    } else if (strcmp(config.alloc, "diag") == 0) {
        g_rs_manager.merge_threshold = 85;   // 对角线批量（提高阈值）
    } else if (strcmp(config.alloc, "sequential") == 0) {
        g_rs_manager.merge_threshold = 85;   // 顺序（提高阈值）
    }
    printf("RS合并阈值: k=%d自适应阈值%d%%\n", config.k, g_rs_manager.merge_threshold);
    g_rs_manager.merge_thread_running = 0;
    
    // 仅走设备侧预留区，不再创建文件式RS
    g_rs_manager.rs_fd = -1;
    memset(g_rs_manager.rs_file_path, 0, sizeof(g_rs_manager.rs_file_path));
    
    g_rs_manager.stripe_metas = (stripe_rs_meta_t*)calloc(total_stripes, sizeof(stripe_rs_meta_t));
    if (!g_rs_manager.stripe_metas) {
        fprintf(stderr, "Failed to allocate RS stripe metadata\n");
        close(g_rs_manager.rs_fd);
        unlink(g_rs_manager.rs_file_path);
        return -1;
    }

// 初始化每个条带的RS元数据
    for (int i = 0; i < total_stripes; i++) {
        stripe_rs_meta_t *meta = &g_rs_manager.stripe_metas[i];
        meta->p_rs_offset = 0;
        meta->q_rs_offset = 0;
        meta->p_rs_size = (uint64_t)RS_AREA_PER_STRIPE;
        meta->q_rs_size = (uint64_t)RS_AREA_PER_STRIPE;
        meta->next_sequence = 1;
        meta->log_count = 0;
        pthread_mutex_init(&meta->p_lock, NULL);
        pthread_mutex_init(&meta->q_lock, NULL);
    }

    // 初始化分片批次：按CPU核数/固定2的幂选择
    if (g_rs_shards == NULL) {
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        if (cores < 1) cores = 1;
        int target = 1;
        while (target < cores) {
            target <<= 1;
            if (target >= 64) { target = 64; break; }
        }
        // 限制到最多 8 个 shard；移除小更新降档，保持按CPU并行
        if (target > 8) target = 8;
        g_rs_shard_count = target;
        g_rs_shards = (rs_shard_t*)calloc((size_t)g_rs_shard_count, sizeof(rs_shard_t));
        if (!g_rs_shards) {
            fprintf(stderr, "Error: RS shard allocation failed\n");
            return -1;
        }
        for (int i = 0; i < g_rs_shard_count; i++) {
            rs_batch_reset(&g_rs_shards[i].batch);
            // 自适应批参数（放大批量与窗口，提高小包聚合效率）
            if (config.update_size <= 8 * 1024) {
                g_rs_shards[i].batch.max_batch = 8192;
                g_rs_shards[i].batch.window_ms = 6;
            } else if (config.update_size <= 16 * 1024) {
                g_rs_shards[i].batch.max_batch = 4096;
                g_rs_shards[i].batch.window_ms = 4;
            } else if (config.update_size <= 32 * 1024) {
                g_rs_shards[i].batch.max_batch = 2048;
                g_rs_shards[i].batch.window_ms = 2;
            } else {
                g_rs_shards[i].batch.max_batch = 1024;
                g_rs_shards[i].batch.window_ms = 2;
            }
            g_rs_shards[i].batch.sort_enabled = 1;
            pthread_mutex_init(&g_rs_shards[i].lock, NULL);
        }
    } else {
        for (int i = 0; i < g_rs_shard_count; i++) {
            rs_batch_reset(&g_rs_shards[i].batch);
        }
    }

    // 兼容单批次（保留但不作为主要路径）
    pthread_mutex_lock(&g_rs_batch_lock);
    rs_batch_reset(&g_rs_batch);
    pthread_mutex_unlock(&g_rs_batch_lock);

    if (!rs_delta_pool) {
        rs_delta_pool = create_enhanced_memory_pool(512, (size_t)config.packetsize, -1, 64);
        if (!rs_delta_pool) {
            fprintf(stderr, "Warning: Failed to create RS delta pool, falling back to malloc/free.\n");
        }
    }
    
    // 初始化RS专用 io_uring（用于批量冲刷）
    if (!g_rs_ring_ready) {
        if (io_uring_queue_init(1024, &g_rs_ring, 0) == 0) {
            g_rs_ring_ready = 1;
        } else {
            g_rs_ring_ready = 0;
        }
    }

    // 启动后台合并线程（仅当具备真实设备固定句柄时）
    if (rs_fixed_io_ready()) {
        g_rs_manager.merge_thread_running = 1;
        if (pthread_create(&g_rs_manager.merge_thread, NULL, rs_merge_thread_func, NULL) != 0) {
            fprintf(stderr, "Failed to create RS merge thread\n");
            g_rs_manager.merge_thread_running = 0;
        }
    } else {
        g_rs_manager.merge_thread_running = 0;
        fprintf(stderr, "[RS] Fixed-file handles not ready; disabling background merge thread.\n");
    }
    
    printf("RS管理器初始化完成: %d个条带, 每条带P/Q各%lluKB RS空间, %s\n", 
           total_stripes, (unsigned long long)(RS_AREA_PER_STRIPE / 1024ULL),
           g_rs_manager.merge_thread_running ? "后台合并已启用" : "后台合并未启用");
    
    return 0;
}

// 销毁RS管理器（优化版本）
static void rs_manager_destroy_optimized(void) {
    // 停止后台合并线程
    if (g_rs_manager.merge_thread_running) {
        g_rs_manager.merge_thread_running = 0;
        pthread_join(g_rs_manager.merge_thread, NULL);
    }

    // 先冲刷所有挂起的批次（在释放条带元数据之前）
    // 1) 分片批次
    if (g_rs_shards) {
        for (int i = 0; i < g_rs_shard_count; i++) {
            rs_batch_t detached;
            int need_flush = 0;
            memset(&detached, 0, sizeof(detached));
            pthread_mutex_lock(&g_rs_shards[i].lock);
            if (g_rs_shards[i].batch.count > 0) {
                rs_batch_detach(&g_rs_shards[i].batch, &detached);
                need_flush = 1;
            }
            pthread_mutex_unlock(&g_rs_shards[i].lock);
        if (need_flush) {
        rs_batch_flush(&detached, g_rs_ring_ready ? &g_rs_ring : NULL);
                rs_batch_free_arrays(&detached);
            }
        }
    }

    // 2) 兼容单批次
    rs_flush_pending();

    // 仅释放条带元数据（无文件映射）
    
    if (g_rs_manager.stripe_metas) {
        for (int i = 0; i < g_rs_manager.total_stripes; i++) {
            pthread_mutex_destroy(&g_rs_manager.stripe_metas[i].p_lock);
            pthread_mutex_destroy(&g_rs_manager.stripe_metas[i].q_lock);
        }
        free(g_rs_manager.stripe_metas);
        g_rs_manager.stripe_metas = NULL;
    }

    // 无文件可关闭/删除（设备侧预留区）

    if (rs_delta_pool) {
        destroy_memory_pool(rs_delta_pool);
        rs_delta_pool = NULL;
    }
    
    // 最后销毁分片锁并释放分片数组
    if (g_rs_shards) {
        for (int i = 0; i < g_rs_shard_count; i++) {
            pthread_mutex_destroy(&g_rs_shards[i].lock);
        }
        free(g_rs_shards);
        g_rs_shards = NULL;
        g_rs_shard_count = 0;
    }

    memset(&g_rs_manager, 0, sizeof(g_rs_manager));

    // 关闭 RS io_uring
    if (g_rs_ring_ready) {
        io_uring_queue_exit(&g_rs_ring);
        g_rs_ring_ready = 0;
    }
}

// ========== PLR-RS 对外API（封装至本地RS管理器） ==========
static void rs_init(int threads) {
    (void)threads; // 当前实现线程数固定由内部控制
    // 估算条带数量：按输入大小或运行时再初始化；此处延迟到主流程传入
    // 在 main 中会调用 rs_manager_init_optimized
}

static void rs_destroy(void) {
    rs_manager_destroy_optimized();
}

static int rs_log_write(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id,
                        uint8_t parity_type, void *delta, uint32_t delta_size) {
    // 环境与参数校验，避免非法访问
    if (!g_rs_manager.rs_enabled) {
        return -1;
    }
    if (!g_rs_manager.stripe_metas || stripe_id >= (uint32_t)g_rs_manager.total_stripes) {
        return -1;
    }
    if (disk_id >= (uint16_t)config.k || packet_id >= (uint16_t)config.w) {
        return -1;
    }
    if (parity_type > 1) {
        return -1;
    }
    if (!delta || delta_size == 0) {
        return 0;
    }
    if (delta_size > (uint32_t)config.packetsize || delta_size > (uint32_t)RS_LOG_ENTRY_MAX_SIZE) {
        return -1;
    }
    // 若既无 mmap 也无有效 fd 且无固定设备句柄，则跳过实际写入（安全返回）
    if (!g_rs_mmap && g_rs_manager.rs_fd < 0 && !rs_fixed_io_ready()) { return 0; }

    // 锁内仅执行轻量追加与“批次分离”
    int append_ok = 0;
    int need_flush = 0;
    rs_batch_t detached;
    memset(&detached, 0, sizeof(detached));

    // 优先分片批次
    int shard_idx = rs_pick_shard(stripe_id);
    if (shard_idx >= 0) {
        rs_shard_t *sh = &g_rs_shards[shard_idx];
        pthread_mutex_lock(&sh->lock);
        if (rs_batch_append(&sh->batch, stripe_id, disk_id, packet_id, parity_type,
                            delta, delta_size, NULL) == 0) {
            append_ok = 1;
            if (rs_batch_should_flush(&sh->batch)) {
                rs_batch_detach(&sh->batch, &detached);
                need_flush = 1;
            }
        }
        pthread_mutex_unlock(&sh->lock);
    } else {
        // 兼容路径：单批次
        pthread_mutex_lock(&g_rs_batch_lock);
        if (rs_batch_append(&g_rs_batch, stripe_id, disk_id, packet_id, parity_type,
                            delta, delta_size, NULL) == 0) {
            append_ok = 1;
            if (rs_batch_should_flush(&g_rs_batch)) {
                rs_batch_detach(&g_rs_batch, &detached);
                need_flush = 1;
            }
        }
        pthread_mutex_unlock(&g_rs_batch_lock);
    }

    if (!append_ok) {
        return -1;
    }

    // 锁外冲刷，避免长时间持锁
            if (need_flush) {
                rs_batch_flush(&detached, g_rs_ring_ready ? &g_rs_ring : NULL);
        rs_batch_free_arrays(&detached);
    }
    return 0;
}

// 计算条带在设备上的RS起始偏移
static off_t rs_get_stripe_base_offset_optimized(int stripe_id, int parity_type) {
    // 设备侧预留区线性布局（对齐4096）：base + stripe * area
    off_t base = (parity_type == 0) ? P_RS_BASE_OFFSET : Q_RS_BASE_OFFSET;
    off_t off = base + (off_t)stripe_id * RS_AREA_PER_STRIPE;
    off = (off + 4095) & ~(off_t)4095; // 4K 对齐
    return off;
}

// 初始化RS内存映射
static int rs_mmap_init(void) {
    // 设备侧路径不使用文件映射
    return -1;
}

// 销毁RS内存映射
static void rs_mmap_destroy(void) {
    // 无文件映射需要销毁
}

// 批量写入多个日志条目到RS（优化版：内存映射+批量优化）
static int rs_write_log_entries_batch(int stripe_id, int *disk_ids, int *packet_ids,
                                     uint8_t *parity_types, void **delta_datas,
                                     uint32_t *delta_sizes, int entry_count,
                                     struct io_uring *ring, int *fds __attribute__((unused))) {
    if (!g_rs_manager.rs_enabled || stripe_id >= g_rs_manager.total_stripes) {
        return -1;
    }
    
    stripe_rs_meta_t *meta = &g_rs_manager.stripe_metas[stripe_id];
    
    int success_count = 0;
    struct timespec io_start, io_end;
    clock_gettime(CLOCK_MONOTONIC, &io_start);

    if (0 && g_rs_mmap) {
        // 先在锁内为每条目预留偏移与序列号，再在锁外执行 memcpy
        off_t *reserved_offsets = (off_t*)alloca(sizeof(off_t) * (size_t)entry_count);
        uint64_t *seqs = (uint64_t*)alloca(sizeof(uint64_t) * (size_t)entry_count);
        for (int i = 0; i < entry_count; i++) {
            uint8_t parity_type = parity_types[i];
            uint64_t entry_total_size = (uint64_t)sizeof(rs_log_entry_t) + (uint64_t)delta_sizes[i];
            uint64_t write_size = (entry_total_size + 4095ULL) & ~4095ULL; // 4K 对齐
            if (parity_type == 0) {
                pthread_mutex_lock(&meta->p_lock);
                if (meta->p_rs_offset + write_size > meta->p_rs_size) {
                    // 触发提前回放，避免频繁回绕
                    if (meta->p_rs_offset * 100 / (meta->p_rs_size ? meta->p_rs_size : 1) >= 85) {
                        pthread_mutex_unlock(&meta->p_lock);
                        rs_merge_stripe_simple(stripe_id);
                        pthread_mutex_lock(&meta->p_lock);
                    }
                    meta->p_rs_offset = 0;
                }
                reserved_offsets[i] = rs_get_stripe_base_offset_optimized(stripe_id, 0) + (off_t)meta->p_rs_offset;
                meta->p_rs_offset += write_size;
                pthread_mutex_unlock(&meta->p_lock);
            } else {
                pthread_mutex_lock(&meta->q_lock);
                if (meta->q_rs_offset + write_size > meta->q_rs_size) {
                    if (meta->q_rs_offset * 100 / (meta->q_rs_size ? meta->q_rs_size : 1) >= 85) {
                        pthread_mutex_unlock(&meta->q_lock);
                        rs_merge_stripe_simple(stripe_id);
                        pthread_mutex_lock(&meta->q_lock);
                    }
                    meta->q_rs_offset = 0;
                }
                reserved_offsets[i] = rs_get_stripe_base_offset_optimized(stripe_id, 1) + (off_t)meta->q_rs_offset;
                meta->q_rs_offset += write_size;
                pthread_mutex_unlock(&meta->q_lock);
            }
            seqs[i] = __atomic_fetch_add(&meta->next_sequence, 1ULL, __ATOMIC_RELAXED);
            meta->log_count++;
            // 置位被触及的packet位图
            if (packet_ids[i] < 512) {
                unsigned idx = (unsigned)(packet_ids[i] / 64);
                unsigned bit = (unsigned)(packet_ids[i] % 64);
                meta->packet_bitmap[idx] |= (1ULL << bit);
            }
        }

        for (int i = 0; i < entry_count; i++) {
            off_t mmap_offset = reserved_offsets[i];
            uint64_t entry_total_size = (uint64_t)sizeof(rs_log_entry_t) + (uint64_t)delta_sizes[i];
            uint64_t write_size = entry_total_size;
            if ((off_t)(mmap_offset + write_size) <= (off_t)g_rs_mmap_size) {
                rs_log_entry_t entry = {
                    .magic = 0x5253504C,
                    .stripe_id = stripe_id,
                    .disk_id = disk_ids[i],
                    .packet_id = packet_ids[i],
                    .delta_size = delta_sizes[i],
                    .sequence = seqs[i],
                    .checksum = 0,
                    .parity_type = parity_types[i]
                };
                char *write_ptr = (char*)g_rs_mmap + mmap_offset;
                memcpy(write_ptr, &entry, sizeof(entry));
                memcpy(write_ptr + sizeof(entry), delta_datas[i], delta_sizes[i]);
                size_t pad = (size_t)write_size - (size_t)entry_total_size;
                if (pad > 0) memset(write_ptr + entry_total_size, 0, pad);
                success_count++;
            }
        }
    } else {
        // 设备侧写入：批内按 parity 聚合为两次写（P一次、Q一次），显著降低 syscall 数
        const size_t hdr_sz = sizeof(rs_log_entry_t);
        // 统计每个 parity 的条目数与总写入字节（4K对齐）
        int cntP = 0, cntQ = 0;
        uint64_t totalP = 0, totalQ = 0;
        for (int i = 0; i < entry_count; i++) {
            uint64_t ent = (uint64_t)hdr_sz + (uint64_t)delta_sizes[i];
            uint64_t aligned = (ent + 4095ULL) & ~4095ULL;
            if (parity_types[i] == 0) { cntP++; totalP += aligned; }
            else { cntQ++; totalQ += aligned; }
        }

        // 为 P/Q 预留连续区间（仅在锁内做偏移递增），并分配对齐的 staging buffer
        off_t p_start_off = 0, q_start_off = 0;
        off_t p_base = rs_get_stripe_base_offset_optimized(stripe_id, 0);
        off_t q_base = rs_get_stripe_base_offset_optimized(stripe_id, 1);

        if (totalP) {
            pthread_mutex_lock(&meta->p_lock);
            if (meta->p_rs_offset + totalP > meta->p_rs_size) {
                if (meta->p_rs_offset * 100 / (meta->p_rs_size ? meta->p_rs_size : 1) >= 85) {
                    pthread_mutex_unlock(&meta->p_lock);
                    rs_merge_stripe_simple(stripe_id);
                    pthread_mutex_lock(&meta->p_lock);
                }
                meta->p_rs_offset = 0;
            }
            p_start_off = (off_t)meta->p_rs_offset;
            meta->p_rs_offset += totalP;
            pthread_mutex_unlock(&meta->p_lock);
        }
        if (totalQ) {
            pthread_mutex_lock(&meta->q_lock);
            if (meta->q_rs_offset + totalQ > meta->q_rs_size) {
                if (meta->q_rs_offset * 100 / (meta->q_rs_size ? meta->q_rs_size : 1) >= 85) {
                    pthread_mutex_unlock(&meta->q_lock);
                    rs_merge_stripe_simple(stripe_id);
                    pthread_mutex_lock(&meta->q_lock);
                }
                meta->q_rs_offset = 0;
            }
            q_start_off = (off_t)meta->q_rs_offset;
            meta->q_rs_offset += totalQ;
            pthread_mutex_unlock(&meta->q_lock);
        }

        // 批量为本次条目分配序列号范围
        uint64_t start_seq = __atomic_fetch_add(&meta->next_sequence, (uint64_t)entry_count, __ATOMIC_RELAXED);
        meta->log_count += (uint64_t)entry_count;

        for (int i = 0; i < entry_count; i++) {
            if (packet_ids[i] < 512) {
                unsigned idx = (unsigned)(packet_ids[i] / 64);
                unsigned bit = (unsigned)(packet_ids[i] % 64);
                meta->packet_bitmap[idx] |= (1ULL << bit);
            }
        }

        // 分配对齐的 staging buffer 并填充 {header, delta} 序列
        void *bufP = NULL, *bufQ = NULL;
        if (totalP) {
            int rcP = posix_memalign(&bufP, 4096, (size_t)totalP);
            if (rcP != 0) { return -1; }
        }
        if (totalQ) {
            int rcQ = posix_memalign(&bufQ, 4096, (size_t)totalQ);
            if (rcQ != 0) { if (bufP) free(bufP); return -1; }
        }
        size_t offP = 0, offQ = 0;
        for (int i = 0; i < entry_count; i++) {
            uint32_t dsz = delta_sizes[i];
            uint64_t ent = (uint64_t)hdr_sz + (uint64_t)dsz;
            uint64_t aligned = (ent + 4095ULL) & ~4095ULL;
            rs_log_entry_t hdr;
            hdr.magic = 0x5253504C;
            hdr.stripe_id = (uint32_t)stripe_id;
            hdr.disk_id = (uint16_t)disk_ids[i];
            hdr.packet_id = (uint16_t)packet_ids[i];
            hdr.delta_size = dsz;
            hdr.sequence = (uint32_t)((start_seq + (uint64_t)i) & 0xffffffffu);
            hdr.checksum = 0;
            hdr.parity_type = parity_types[i];
            if (parity_types[i] == 0) {
                memcpy((char*)bufP + offP, &hdr, hdr_sz);
                memcpy((char*)bufP + offP + hdr_sz, delta_datas[i], dsz);
                size_t pad = (size_t)aligned - (size_t)ent;
                if (pad) memset((char*)bufP + offP + (size_t)ent, 0, pad);
                offP += (size_t)aligned;
            } else {
                memcpy((char*)bufQ + offQ, &hdr, hdr_sz);
                memcpy((char*)bufQ + offQ + hdr_sz, delta_datas[i], dsz);
                size_t pad = (size_t)aligned - (size_t)ent;
                if (pad) memset((char*)bufQ + offQ + (size_t)ent, 0, pad);
                offQ += (size_t)aligned;
            }
        }

        int fd_p = (g_fixed.handles ? g_fixed.handles[config.k + 0] : -1);
        int fd_q = (g_fixed.handles ? g_fixed.handles[config.k + 1] : -1);

        // 写入：优先 io_uring 单次写；否则 pwrite 单次
        if (ring && g_fixed.enabled) {
            int submits = 0;
            if (totalP && fd_p >= 0) {
                struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
                if (sqe) {
                    io_uring_prep_write(sqe, fd_p, bufP, (unsigned)totalP, p_base + p_start_off);
                    io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                    submits++;
                }
            }
            if (totalQ && fd_q >= 0) {
                struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
                if (sqe) {
                    io_uring_prep_write(sqe, fd_q, bufQ, (unsigned)totalQ, q_base + q_start_off);
                    io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
                    submits++;
                }
            }
            if (submits) {
                (void)io_uring_submit(ring);
                struct io_uring_cqe *cqe;
                for (int c = 0; c < submits; c++) {
                    if (io_uring_wait_cqe(ring, &cqe) < 0) break;
                    if (cqe->res >= 0) success_count += (entry_count > 0 ? entry_count : 0); // 计入条目成功数
                    io_uring_cqe_seen(ring, cqe);
                }
            }
            if (bufP) free(bufP);
            if (bufQ) free(bufQ);
        } else {
            if (totalP && fd_p >= 0) {
                ssize_t w = pwrite(fd_p, bufP, (size_t)totalP, p_base + p_start_off);
                if (w >= 0) success_count += cntP;
            }
            if (totalQ && fd_q >= 0) {
                ssize_t w = pwrite(fd_q, bufQ, (size_t)totalQ, q_base + q_start_off);
                if (w >= 0) success_count += cntQ;
            }
            if (bufP) free(bufP);
            if (bufQ) free(bufQ);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &io_end);
    if (success_count > 0) {
        record_io_events(timespec_diff_sec(&io_start, &io_end), success_count, 1);
    }
    
    return success_count;
}

// 检查条带是否需要合并
static __attribute__((unused)) int rs_should_merge_stripe(int stripe_id) {
    if (!g_rs_manager.rs_enabled || stripe_id >= g_rs_manager.total_stripes) {
        return 0;
    }
    
    stripe_rs_meta_t *meta = &g_rs_manager.stripe_metas[stripe_id];
    int p_usage = (meta->p_rs_offset * 100) / meta->p_rs_size;
    int q_usage = (meta->q_rs_offset * 100) / meta->q_rs_size;
    
    return (p_usage >= g_rs_manager.merge_threshold) || 
           (q_usage >= g_rs_manager.merge_threshold);
}

// 简化的条带合并（基础版本）
static int rs_merge_stripe_simple(int stripe_id) {
    // 切换到具备“日志回放到基线”的真实合并实现
    return rs_merge_stripe_replay(stripe_id);
}

// 读取并回放某个条带的 RS 日志到基线 P/Q，然后写回校验盘
static int rs_merge_stripe_replay(int stripe_id) {
    if (!g_rs_manager.rs_enabled || stripe_id < 0 || stripe_id >= g_rs_manager.total_stripes) {
        return -1;
    }

    // 若未准备好真实设备句柄，则直接返回，不做任何回放以避免无谓争用
    if (!rs_fixed_io_ready()) {
        return 0;
    }

    stripe_rs_meta_t *meta = &g_rs_manager.stripe_metas[stripe_id];
pthread_mutex_lock(&meta->p_lock);
pthread_mutex_lock(&meta->q_lock);

    const size_t stripe_size = (size_t)config.packetsize * (size_t)config.w;
    const off_t stripe_offset = (off_t)stripe_id * (off_t)stripe_size;

    if (meta->p_rs_offset == 0 && meta->q_rs_offset == 0) {
pthread_mutex_unlock(&meta->q_lock);
pthread_mutex_unlock(&meta->p_lock);
        return 0; // 无日志可合并
    }

    if (config.n_updates <= 100) {
        printf("回放条带%d的RS日志: P:%llu字节, Q:%llu字节\n", stripe_id, (unsigned long long)meta->p_rs_offset, (unsigned long long)meta->q_rs_offset);
    }

// 分配基线 P/Q 缓冲并读取当前校验
    char *par_p = (char*)pool_alloc(global_pool);
    char *par_q = (char*)pool_alloc(global_pool);
            if (!par_p || !par_q) {
        if (par_p) pool_free(global_pool, par_p);
        if (par_q) pool_free(global_pool, par_q);
                pthread_mutex_unlock(&meta->q_lock);
                pthread_mutex_unlock(&meta->p_lock);
        return -1;
    }

// 使用一个临时 io_uring 实例用于并发读写校验块
    struct io_uring ring_local;
    if (io_uring_queue_init(256, &ring_local, 0) < 0) {
        pool_free(global_pool, par_p);
        pool_free(global_pool, par_q);
        pthread_mutex_unlock(&meta->q_lock);
        pthread_mutex_unlock(&meta->p_lock);
        return -1;
    }

    // 读取当前 P/Q：仅按位图读取被触及的packet切片
    for (int b = 0; b < 8 && (b * 64) < config.w; b++) {
        uint64_t bm = meta->packet_bitmap[b];
        while (bm) {
            int t = __builtin_ctzll(bm);
            int pkt = b * 64 + t;
            if (pkt < config.w) {
                size_t offb = (size_t)pkt * (size_t)config.packetsize;
                int fd_p = g_fixed.handles[config.k + 0];
                int fd_q = g_fixed.handles[config.k + 1];
                if (fd_p >= 0) { ssize_t rr = pread(fd_p, par_p + offb, (size_t)config.packetsize, stripe_offset + (off_t)offb); (void)rr; }
                if (fd_q >= 0) { ssize_t rr2 = pread(fd_q, par_q + offb, (size_t)config.packetsize, stripe_offset + (off_t)offb); (void)rr2; }
            }
            bm &= (bm - 1);
        }
    }

// 应用 P 的日志（C 实现）
    {
        int parity_type = 0;
        off_t area_base = rs_get_stripe_base_offset_optimized(stripe_id, parity_type);
        int fd_dev = (g_fixed.handles ? g_fixed.handles[config.k + 0] : -1);
        uint64_t used = meta->p_rs_offset;
        if (used > 0) {
            char *buf = (char*)malloc((size_t)used);
            if (!buf) { io_uring_queue_exit(&ring_local); pool_free(global_pool, par_p); pool_free(global_pool, par_q); pthread_mutex_unlock(&meta->q_lock); pthread_mutex_unlock(&meta->p_lock); return -1; }
            ssize_t r = pread(fd_dev, buf, (size_t)used, area_base);
            if (r != (ssize_t)used) { free(buf); io_uring_queue_exit(&ring_local); pool_free(global_pool, par_p); pool_free(global_pool, par_q); pthread_mutex_unlock(&meta->q_lock); pthread_mutex_unlock(&meta->p_lock); return -1; }
            for (size_t pos = 0; pos + sizeof(rs_log_entry_t) <= (size_t)used; ) {
                rs_log_entry_t *hdr = (rs_log_entry_t*)(buf + pos);
                if (hdr->magic != 0x5253504C || hdr->stripe_id != (uint32_t)stripe_id || hdr->parity_type != (uint8_t)parity_type) { break; }
                size_t dsz = hdr->delta_size; size_t offh = pos + sizeof(rs_log_entry_t);
                if (offh + dsz > (size_t)used) break;
                size_t offb = (size_t)hdr->packet_id * (size_t)config.packetsize;
                xor_update_simd(par_p + offb, (char*)(buf + offh), dsz);
                // 对齐推进（与写入时一致）
                size_t step = ((sizeof(rs_log_entry_t) + dsz + 4095U) & ~4095U);
                pos += step;
            }
            free(buf);
        }
    }
    // 应用 Q 的日志（C 实现）
    {
        int parity_type = 1;
        off_t area_base = rs_get_stripe_base_offset_optimized(stripe_id, parity_type);
        int fd_dev = (g_fixed.handles ? g_fixed.handles[config.k + 1] : -1);
        uint64_t used = meta->q_rs_offset;
        if (used > 0) {
            char *buf = (char*)malloc((size_t)used);
            if (!buf) { io_uring_queue_exit(&ring_local); pool_free(global_pool, par_p); pool_free(global_pool, par_q); pthread_mutex_unlock(&meta->q_lock); pthread_mutex_unlock(&meta->p_lock); return -1; }
            ssize_t r = pread(fd_dev, buf, (size_t)used, area_base);
            if (r != (ssize_t)used) { free(buf); io_uring_queue_exit(&ring_local); pool_free(global_pool, par_p); pool_free(global_pool, par_q); pthread_mutex_unlock(&meta->q_lock); pthread_mutex_unlock(&meta->p_lock); return -1; }
            for (size_t pos = 0; pos + sizeof(rs_log_entry_t) <= (size_t)used; ) {
                rs_log_entry_t *hdr = (rs_log_entry_t*)(buf + pos);
                if (hdr->magic != 0x5253504C || hdr->stripe_id != (uint32_t)stripe_id || hdr->parity_type != (uint8_t)parity_type) { break; }
                size_t dsz = hdr->delta_size; size_t offh = pos + sizeof(rs_log_entry_t);
                if (offh + dsz > (size_t)used) break;
                size_t offb = (size_t)hdr->packet_id * (size_t)config.packetsize;
                xor_update_simd(par_q + offb, (char*)(buf + offh), dsz);
                size_t step = ((sizeof(rs_log_entry_t) + dsz + 4095U) & ~4095U);
                pos += step;
            }
            free(buf);
        }
    }

    // 将更新后的 P/Q 写回校验盘：合并连续 packet 区间一次写入
    {
        int fd_p = g_fixed.handles ? g_fixed.handles[config.k + 0] : -1;
        int fd_q = g_fixed.handles ? g_fixed.handles[config.k + 1] : -1;
        // 扫描 [0..w-1] 的连续1区间
        int pkt = 0;
        while (pkt < config.w) {
            // 查找下一个被触及的packet
            int start = -1;
            for (int p = pkt; p < config.w; p++) {
                int idx = p / 64, bit = p % 64;
                if (meta->packet_bitmap[idx] & (1ULL << bit)) { start = p; break; }
            }
            if (start < 0) break;
            int end = start;
            // 扩展连续段
            for (int p = start + 1; p < config.w; p++) {
                int idx = p / 64, bit = p % 64;
                if (meta->packet_bitmap[idx] & (1ULL << bit)) end = p; else break;
            }
            size_t offb = (size_t)start * (size_t)config.packetsize;
            size_t bytes = (size_t)(end - start + 1) * (size_t)config.packetsize;
            if (fd_p >= 0) {
                ssize_t w1 = pwrite(fd_p, par_p + offb, bytes, stripe_offset + (off_t)offb);
                if (w1 != (ssize_t)bytes) {
                    fprintf(stderr, "P writeback range [%d,%d] failed: %s\n", start, end, strerror(errno));
                }
            }
            if (fd_q >= 0) {
                ssize_t w2 = pwrite(fd_q, par_q + offb, bytes, stripe_offset + (off_t)offb);
                if (w2 != (ssize_t)bytes) {
                    fprintf(stderr, "Q writeback range [%d,%d] failed: %s\n", start, end, strerror(errno));
                }
            }
            pkt = end + 1;
        }
    }

    io_uring_queue_exit(&ring_local);

    // 清理并复位 RS 区元数据
    pool_free(global_pool, par_p);
    pool_free(global_pool, par_q);

    meta->p_rs_offset = 0;
    meta->q_rs_offset = 0;
    meta->log_count = 0;
    for (int b = 0; b < 8; b++) meta->packet_bitmap[b] = 0ULL;

    pthread_mutex_unlock(&meta->q_lock);
    pthread_mutex_unlock(&meta->p_lock);
    return 0;
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

// 新增：对所有设备并行 fsync，确保数据落盘
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
void update_evenodd_simple(char **data __attribute__((unused)), char **coding __attribute__((unused)), char *new_data,
                           int *fds __attribute__((unused)), int total_disks __attribute__((unused)), struct io_uring *ring,
                           long stripe_num, long *next_stripe_cursor) {
    int blocks_to_update = config.update_size / config.packetsize;
    int blocks_per_stripe = (config.k > 0 && config.w > 0) ? config.k * config.w : 0;
    
    struct timespec upd_start_ts, upd_end_ts;
    clock_gettime(CLOCK_MONOTONIC, &upd_start_ts);
    
    if (config.n_updates <= 20) {
        printf("深度集成更新: %d块 (策略=%s, RS=%s, PARIX=%s)\n",
               blocks_to_update, config.alloc,
               config.use_rs ? "ON" : "OFF",
               config.use_parix ? "ON" : "OFF");
    }

    

    // 共享规划器：按策略/RS倾向生成计划
    BlockPos *allocation_plan = NULL;
    const char *plan_used = NULL;
    int plan_count = 0;
    int need_plan = blocks_to_update > 0 &&
        ((strcmp(((config.alloc && config.alloc[0])?config.alloc:"sequential"), "sequential") != 0) ||
         (config.use_rs && g_rs_manager.rs_enabled) ||
         (config.use_parix && g_parix_manager.parix_enabled));

    if (need_plan) {
        allocation_plan = (BlockPos*)malloc(sizeof(BlockPos) * (size_t)blocks_to_update);
        if (allocation_plan) {
            plan_count = plan_block_positions(blocks_to_update, allocation_plan,
                                              config.use_rs && g_rs_manager.rs_enabled,
                                              &plan_used);
            if (plan_count != blocks_to_update) {
                fill_sequential_plan(allocation_plan, blocks_to_update);
                plan_count = blocks_to_update;
                plan_used = "sequential";
            }
        }
    }
    int plan_valid = allocation_plan && plan_count == blocks_to_update;

    // 预计算每块的条带ID
    uint32_t *stripe_ids = NULL;
    long stripes_total = (stripe_num > 0) ? stripe_num : 1;
    if (blocks_to_update > 0 &&
        ((config.use_rs && g_rs_manager.rs_enabled) ||
         (config.use_parix && g_parix_manager.parix_enabled))) {
        stripe_ids = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)blocks_to_update);
        if (stripe_ids) {
            for (int i = 0; i < blocks_to_update; i++) {
                uint32_t sid;
                if (strcmp(config.mode, "random") == 0) {
                    sid = (uint32_t)(rand() % stripes_total);
                } else if (blocks_per_stripe > 0) {
                    long stripe_offset = i / blocks_per_stripe;
                    sid = (uint32_t)((*next_stripe_cursor + stripe_offset) % stripes_total);
                } else {
                    sid = 0;
                }
                stripe_ids[i] = sid;
            }
        }
    }

    // RS：按计划直接写入（避免额外排序开销）
    if (config.use_rs && g_rs_manager.rs_enabled && rs_fixed_io_ready()) {
        if (plan_valid && plan_used) {
            printf("RS planner: %s mapping (%d blocks)\n", plan_used, plan_count);
        }
        for (int i = 0; i < blocks_to_update; i++) {
            uint32_t stripe_id = stripe_ids ? stripe_ids[i]
                                   : (strcmp(config.mode, "random") == 0 ? (uint32_t)(rand() % stripes_total)
                                      : (blocks_per_stripe > 0 ? (uint32_t)((*next_stripe_cursor + (i / blocks_per_stripe)) % stripes_total) : 0));
            uint16_t disk_id = plan_valid ? (uint16_t)allocation_plan[i].col : (uint16_t)(i % config.k);
            uint16_t packet_id = plan_valid ? (uint16_t)(allocation_plan[i].row % config.w)
                                            : (uint16_t)((i / config.k) % config.w);
            if (rs_log_write(stripe_id, disk_id, packet_id, 0,
                             new_data + (size_t)i * config.packetsize,
                             (uint32_t)config.packetsize) != 0) {
                printf("RS日志写入失败: stripe=%u, disk=%u, row=%u\n", stripe_id, disk_id, packet_id);
            }
        }
    }

    // PARIX：消费相同计划，条带内按chunk批量更新元数据与写入
    if (config.use_parix && g_parix_manager.parix_enabled) {
        if (plan_valid && plan_used) {
            printf("PARIX planner: %s mapping (%d blocks)\n", plan_used, plan_count);
        }
        int i = 0;
        while (i < blocks_to_update) {
            uint32_t stripe_id = stripe_ids ? stripe_ids[i]
                                   : (strcmp(config.mode, "random") == 0 ? (uint32_t)(rand() % stripes_total)
                                      : (blocks_per_stripe > 0 ? (uint32_t)((*next_stripe_cursor + (i / blocks_per_stripe)) % stripes_total) : 0));
            int chunk = 1;
            if (stripe_ids) {
                int max_chunk = (blocks_per_stripe > 0) ? blocks_per_stripe : blocks_to_update;
                while (i + chunk < blocks_to_update && stripe_ids[i + chunk] == stripe_id && chunk < max_chunk) chunk++;
            } else if (strcmp(config.mode, "random") != 0 && blocks_per_stripe > 0) {
                int logical_idx = i % blocks_per_stripe;
                chunk = blocks_per_stripe - logical_idx;
                if (chunk > blocks_to_update - i) chunk = blocks_to_update - i;
            }

            if (plan_valid) {
                parix_update_allocation_meta(stripe_id, allocation_plan + i, chunk);
            }

            for (int j = 0; j < chunk; j++) {
                int idx = i + j;
                BlockPos *ctx = plan_valid ? &allocation_plan[idx] : NULL;
                uint16_t disk_id = ctx ? (uint16_t)ctx->col : (uint16_t)(idx % config.k);
                uint16_t packet_id = ctx ? (uint16_t)(ctx->row % config.w)
                                         : (uint16_t)((idx / config.k) % config.w);
                void *payload = new_data + (size_t)idx * config.packetsize;

                int rc = parix_speculative_write_enhanced(stripe_id, disk_id, packet_id,
                                                          payload, (uint32_t)config.packetsize, ctx);
                if (rc < 0) {
                    if (parix_speculative_write(stripe_id, disk_id, packet_id,
                                                payload, (uint32_t)config.packetsize) != 0) {
                        printf("PARIX推测写入失败: stripe=%u, disk=%u, row=%u\n",
                               stripe_id, disk_id, packet_id);
                    }
                }
            }
            i += chunk;
        }
    }

    free(stripe_ids);
    free(allocation_plan);
    
    clock_gettime(CLOCK_MONOTONIC, &upd_end_ts);
    double elapsed = timespec_diff_sec(&upd_start_ts, &upd_end_ts);
    
    pthread_mutex_lock(&stats.lock);
    stats.compute_time += elapsed;
    stats.update_count++;
    pthread_mutex_unlock(&stats.lock);
}

// PARIX后台重放线程函数
static void* parix_replay_thread_func(void *arg) {
    (void)arg;
    const int threshold_entries = 1024;
    while (g_parix_manager.replay_thread_running) {
        // 条件等待：50ms 超时或收到唤醒
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        long ns = ts.tv_nsec + 50L * 1000L * 1000L; // 50ms
        ts.tv_sec += ns / 1000000000L; ts.tv_nsec = ns % 1000000000L;
        pthread_mutex_lock(&g_parix_manager.replay_mu);
        (void)pthread_cond_timedwait(&g_parix_manager.replay_cv, &g_parix_manager.replay_mu, &ts);
        g_parix_manager.need_replay = 0;
        pthread_mutex_unlock(&g_parix_manager.replay_mu);

        for (int stripe_id = 0; stripe_id < g_parix_manager.total_stripes; stripe_id++) {
            if (!g_parix_manager.replay_thread_running) break;
            stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
            pthread_mutex_lock(&meta->parix_lock);
            int usage = (int)((meta->parix_offset * 100ULL) / (meta->parix_size ? meta->parix_size : 1));
            int cnt = meta->parix_count;
            pthread_mutex_unlock(&meta->parix_lock);
            if (cnt >= threshold_entries || usage >= 95) {
                parix_replay_stripe((uint32_t)stripe_id);
            }
        }
    }
    return NULL;
}

// 智能推测策略：基于地址分配模式判断是否应该推测
static __attribute__((unused)) int parix_should_speculate(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return 0;
    }

// 边界检查：防止数组越界
    if (disk_id >= 16 || packet_id >= 16) {
        return 0;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    
    // 检查该数据块是否已经处于推测状态
    if (meta->data_blocks[disk_id][packet_id].is_speculative) {
        return 1; // 继续推测
    }

// 智能推测：基于地址分配模式调整推测策略
    int adaptive_threshold = parix_adaptive_threshold(stripe_id, meta->allocation_meta.current_allocation_strategy);
    // （移除无效引用：该函数不接受分配上下文）
    
    // 检查连续写入次数是否达到自适应阈值
    if (meta->data_blocks[disk_id][packet_id].write_count >= (uint32_t)adaptive_threshold) {
        return 1; // 开始推测
    }

// 基于地址分配模式的额外推测条件
    if (meta->allocation_meta.allocation_efficiency > 0.8) {
        // 高分配效率时，降低推测阈值
        if (meta->data_blocks[disk_id][packet_id].write_count >= (uint32_t)(adaptive_threshold / 2)) {
            return 1;
        }
    }
    
    return 0; // 不推测
}

// 分析地址分配模式，为PARIX推测提供智能指导
static int parix_analyze_allocation_pattern(uint32_t stripe_id, BlockPos *blocks, int block_count) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return -1;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    
    // 分析分配模式
    int row_count = 0, diag_count = 0, mixed_count __attribute__((unused)) = 0;
    int row_usage[16] = {0}, diag_usage[16] = {0};
    
    for (int i = 0; i < block_count; i++) {
        int row = blocks[i].row;
        int diag = blocks[i].diag;
        
        if (row >= 0 && row < 16) {
            if (!row_usage[row]) {
                row_usage[row] = 1;
                row_count++;
            }
        }
        
        if (diag >= 0 && diag < 16) {
            if (!diag_usage[diag]) {
                diag_usage[diag] = 1;
                diag_count++;
            }
        }
    }

// 判断分配策略类型
    if (row_count > diag_count * 1.5) {
        meta->allocation_meta.current_allocation_strategy = 0; // 行优先
    } else if (diag_count > row_count * 1.5) {
        meta->allocation_meta.current_allocation_strategy = 1; // 对角线优先
    } else {
        meta->allocation_meta.current_allocation_strategy = 2; // 混合策略
    }

// 计算分配效率
    int total_unique = row_count + diag_count;
    meta->allocation_meta.allocation_efficiency = (double)total_unique / (block_count * 2);
    
    // 更新统计信息
    meta->allocation_meta.row_blocks_count = row_count;
    meta->allocation_meta.diag_blocks_count = diag_count;
    meta->allocation_meta.mixed_blocks_count = block_count - row_count - diag_count;
    
    // 识别热点行和对角线
    for (int i = 0; i < 16; i++) {
        meta->allocation_meta.hotspot_rows[i] = row_usage[i];
        meta->allocation_meta.hotspot_diags[i] = diag_usage[i];
    }
    
    if (PARIX_DEBUG_LOG) {
        printf("PARIX分配模式分析: 条带%d, 行%d, 对角线%d, 效率%.2f, 策略%d\n",
               stripe_id, row_count, diag_count, meta->allocation_meta.allocation_efficiency,
               meta->allocation_meta.current_allocation_strategy);
    }
    
    return 0;
}

// 智能推测：基于地址分配上下文进行推测决策
static int parix_smart_speculation(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id, 
                                   BlockPos *allocation_context) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return 0;
    }

// 边界检查：防止数组越界
    if (disk_id >= 16 || packet_id >= 16) {
        return 0;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    
    // 基于地址分配模式的推测优先级计算
    int speculation_priority = 0;
    
    // 检查是否在热点行或对角线
    if (allocation_context) {
        int row = allocation_context->row;
        int diag = allocation_context->diag;
        
        if (row >= 0 && row < 16 && meta->allocation_meta.hotspot_rows[row]) {
            speculation_priority += 3; // 热点行加分
        }
        
        if (diag >= 0 && diag < 16 && meta->allocation_meta.hotspot_diags[diag]) {
            speculation_priority += 3; // 热点对角线加分
        }
    }

// 基于分配效率调整推测策略
    if (meta->allocation_meta.allocation_efficiency > 0.7) {
        speculation_priority += 2; // 高分配效率加分
    }

// 基于连续写入模式调整
    speculation_priority += meta->data_blocks[disk_id][packet_id].write_count;
    
    // 更新推测优先级
    meta->data_blocks[disk_id][packet_id].speculation_priority = speculation_priority;
    
    // 智能推测决策
    int adaptive_threshold = parix_adaptive_threshold(stripe_id, meta->allocation_meta.current_allocation_strategy);
    
    if (speculation_priority >= adaptive_threshold) {
        return 1; // 进行推测
    }
    
    return 0; // 不推测
}

// 自适应推测阈值：根据地址分配策略调整
static int parix_adaptive_threshold(uint32_t stripe_id __attribute__((unused)), int allocation_strategy) {
    int base_threshold;
    // 根据更新大小设置基础阈值：4K→1，8–16K→2，≥32K→3
    if (config.update_size <= 4*1024) base_threshold = 1;
    else if (config.update_size <= 16*1024) base_threshold = 2;
    else base_threshold = 3;

    // 根据地址分配策略微调
    switch (allocation_strategy) {
        case 0: // 行优先分配
            base_threshold -= 1; // 行分配更连续，降低阈值
            break;
        case 1: // 对角线优先分配
            // 保持
            break;
        case 2: // 混合策略
            base_threshold += 1; // 混合更复杂，适度提高
            break;
        default:
            break;
    }
    if (base_threshold < 1) base_threshold = 1;
    return base_threshold;
}

// 更新地址分配元数据
static void parix_update_allocation_meta(uint32_t stripe_id, BlockPos *blocks, int block_count) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    
    // 分析当前分配模式
    parix_analyze_allocation_pattern(stripe_id, blocks, block_count);
    
    // 为每个数据块设置地址分配提示
    for (int i = 0; i < block_count; i++) {
        int disk_id = blocks[i].col;
        int packet_id = blocks[i].row;
        
        if (disk_id >= 0 && disk_id < 16 && packet_id >= 0 && packet_id < 16) {
            // 设置地址分配提示
            if (meta->allocation_meta.current_allocation_strategy == 0) {
                meta->data_blocks[disk_id][packet_id].allocation_hint = 0; // 行分配
            } else if (meta->allocation_meta.current_allocation_strategy == 1) {
                meta->data_blocks[disk_id][packet_id].allocation_hint = 1; // 对角线分配
            } else {
                meta->data_blocks[disk_id][packet_id].allocation_hint = 2; // 混合分配
            }
        }
    }
}

// 预测推测性写入的收益
static int parix_predict_speculation_benefit(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return 0;
    }

// 边界检查：防止数组越界
    if (disk_id >= 16 || packet_id >= 16) {
        return 0;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    
    // 基于历史写入模式预测收益
    int write_count = meta->data_blocks[disk_id][packet_id].write_count;
    int speculation_priority = meta->data_blocks[disk_id][packet_id].speculation_priority;
    
    // 计算预测收益分数 (0-100)
    int benefit_score = 0;
    
    // 连续写入次数收益
    benefit_score += write_count * 10;
    
    // 推测优先级收益
    benefit_score += speculation_priority * 5;
    
    // 地址分配效率收益
    benefit_score += (int)(meta->allocation_meta.allocation_efficiency * 20);
    
    // 热点区域收益
    if (meta->allocation_meta.hotspot_rows[packet_id] || 
        meta->allocation_meta.hotspot_diags[(disk_id + packet_id) % (config.k - 1)]) {
        benefit_score += 15;
    }
    
    return (benefit_score > 50) ? 1 : 0; // 收益分数超过50分才进行推测
}

// PARIX推测性写入
static int parix_speculative_write(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id,
                                   void *new_data, uint32_t data_size) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return -1;
    }

// 边界检查：防止数组越界
    if (disk_id >= 16 || packet_id >= 16) {
        fprintf(stderr, "PARIX警告: disk_id=%u或packet_id=%u超出范围(>=16)，跳过写入\n", disk_id, packet_id);
        return -1;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    pthread_mutex_lock(&meta->parix_lock);
    
    // 构建PARIX日志条目
    parix_log_entry_t entry = {
        .magic = 0x50415249, // 'PARI'
        .stripe_id = stripe_id,
        .disk_id = disk_id,
        .packet_id = packet_id,
        .data_size = data_size,
        .sequence = meta->parix_sequence++,
        .checksum = 0,
        .is_initial = !meta->data_blocks[disk_id][packet_id].is_speculative
    };
    
    uint32_t entry_total_size = sizeof(parix_log_entry_t) + data_size;
    
    // 检查空间是否足够
    if (meta->parix_offset + entry_total_size > meta->parix_size) {
        // 空间不足，触发重放
        pthread_mutex_unlock(&meta->parix_lock);
        parix_replay_stripe(stripe_id);
        pthread_mutex_lock(&meta->parix_lock);
        meta->parix_offset = 0; // 重置偏移
    }

    // 写入PARIX日志（设备侧：P盘预留区）
    off_t write_offset = PARIX_BASE_OFFSET + (off_t)stripe_id * PARIX_AREA_PER_STRIPE + meta->parix_offset;

    int fd_parix = (g_fixed.handles ? g_fixed.handles[config.k + 0] : g_parix_manager.parix_fd);
    if (g_parix_ring_ready && g_bufpool.enabled && g_fixed.enabled && fd_parix >= 0) {
        void *ptr = NULL; int idx = bufpool_acquire(&ptr);
        if (idx >= 0 && (size_t)g_bufpool.iovecs[idx].iov_len >= entry_total_size) {
            memcpy(ptr, &entry, sizeof(entry));
            memcpy((char*)ptr + sizeof(entry), new_data, data_size);
            struct io_uring_sqe *sqe = io_uring_get_sqe(&g_parix_ring);
            if (!sqe) { bufpool_release(idx); pthread_mutex_unlock(&meta->parix_lock); return -1; }
            io_uring_prep_write_fixed(sqe, fd_parix, ptr, (unsigned)entry_total_size, write_offset, idx);
            if (g_fixed.enabled) io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            if (io_uring_submit(&g_parix_ring) < 0) { bufpool_release(idx); pthread_mutex_unlock(&meta->parix_lock); return -1; }
            struct io_uring_cqe *cqe; if (io_uring_wait_cqe(&g_parix_ring, &cqe) < 0) { bufpool_release(idx); pthread_mutex_unlock(&meta->parix_lock); return -1; }
            int ok = (cqe->res == (int)entry_total_size);
            io_uring_cqe_seen(&g_parix_ring, cqe);
            bufpool_release(idx);
            if (!ok) { pthread_mutex_unlock(&meta->parix_lock); return -1; }
        } else {
            if (idx >= 0) bufpool_release(idx);
            struct iovec iov[2] = {
                { .iov_base = &entry,   .iov_len = sizeof(entry) },
                { .iov_base = new_data, .iov_len = data_size }
            };
            ssize_t written = pwritev(fd_parix, iov, 2, write_offset);
            if (written != (ssize_t)entry_total_size) { pthread_mutex_unlock(&meta->parix_lock); return -1; }
        }
    } else {
        struct iovec iov[2] = {
            { .iov_base = &entry,   .iov_len = sizeof(entry) },
            { .iov_base = new_data, .iov_len = data_size }
        };
        ssize_t written = pwritev(fd_parix, iov, 2, write_offset);
        if (written != (ssize_t)entry_total_size) { pthread_mutex_unlock(&meta->parix_lock); return -1; }
    }

// 更新元数据
    meta->parix_offset += entry_total_size;
    meta->parix_count++;
    
    // 更新数据块推测状态
    if (entry.is_initial) {
        // 首次写入，需要保存基础数据
        meta->data_blocks[disk_id][packet_id].base_data = malloc(data_size);
        if (meta->data_blocks[disk_id][packet_id].base_data) {
            memcpy(meta->data_blocks[disk_id][packet_id].base_data, new_data, data_size);
        }
        meta->data_blocks[disk_id][packet_id].base_sequence = entry.sequence;
    }
    
    meta->data_blocks[disk_id][packet_id].is_speculative = 1;
    meta->data_blocks[disk_id][packet_id].write_count++;
    clock_gettime(CLOCK_MONOTONIC, &meta->data_blocks[disk_id][packet_id].last_write);
    // 触发条件变量：条数≥1024或使用率≥95%
    {
        int usage_now = (int)((meta->parix_offset * 100ULL) / (meta->parix_size ? meta->parix_size : 1));
        if (meta->parix_count >= 1024 || usage_now >= 95) {
            pthread_mutex_lock(&g_parix_manager.replay_mu);
            g_parix_manager.need_replay = 1;
            pthread_cond_signal(&g_parix_manager.replay_cv);
            pthread_mutex_unlock(&g_parix_manager.replay_mu);
        }
    }
    
    pthread_mutex_unlock(&meta->parix_lock);
    
#if PARIX_DEBUG_LOG
    printf("PARIX推测性写入: 条带%d, 磁盘%d, 包%d, 大小%d, 序列%d, %s\n",
           stripe_id, disk_id, packet_id, data_size, entry.sequence,
           entry.is_initial ? "初始" : "推测");
#endif
    
    return 0;
}

// PARIX条带重放
static int parix_replay_stripe(uint32_t stripe_id) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return -1;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    
    if (PARIX_DEBUG_LOG) {
        printf("PARIX重放条带%d: 开始重放%d个推测性写入\n", stripe_id, meta->parix_count);
    }

    // 1) 扫描日志，收集每个 (disk, packet) 的最终新值
    size_t pkt_sz = (size_t)config.packetsize;
    unsigned char *final_new[16][16];
    memset(final_new, 0, sizeof(final_new));

    off_t base_off = g_parix_manager.parix_base_offset + (off_t)stripe_id * meta->parix_size;
    off_t pos = 0;
    while (pos + (off_t)sizeof(parix_log_entry_t) <= (off_t)meta->parix_offset) {
        parix_log_entry_t hdr;
        ssize_t r1 = pread(g_parix_manager.parix_fd, &hdr, sizeof(hdr), base_off + pos);
        if (r1 != (ssize_t)sizeof(hdr)) break;
        if (hdr.magic != 0x50415249 || hdr.stripe_id != stripe_id) break;
        if (hdr.disk_id >= 16 || hdr.packet_id >= 16) { pos += sizeof(hdr) + hdr.data_size; continue; }
        unsigned char *buf = (unsigned char*)malloc(hdr.data_size);
        if (!buf) { pos += sizeof(hdr) + hdr.data_size; continue; }
        ssize_t r2 = pread(g_parix_manager.parix_fd, buf, hdr.data_size, base_off + pos + (off_t)sizeof(hdr));
        if (r2 != (ssize_t)hdr.data_size) { free(buf); break; }
        // 记录最终新值
        if (final_new[hdr.disk_id][hdr.packet_id]) free(final_new[hdr.disk_id][hdr.packet_id]);
        final_new[hdr.disk_id][hdr.packet_id] = buf;
        pos += (off_t)sizeof(hdr) + (off_t)hdr.data_size;
    }

    // 2) 读取当前 P/Q
    const size_t stripe_size = pkt_sz * (size_t)config.w;
    const off_t stripe_offset = (off_t)stripe_id * (off_t)stripe_size;
    char *par_p = (char*)pool_alloc(global_pool);
    char *par_q = (char*)pool_alloc(global_pool);
    if (!par_p || !par_q) { if (par_p) pool_free(global_pool, par_p); if (par_q) pool_free(global_pool, par_q); goto cleanup_only; }
    int fd_p = g_fixed.handles ? g_fixed.handles[config.k + 0] : -1;
    int fd_q = g_fixed.handles ? g_fixed.handles[config.k + 1] : -1;
    if (fd_p >= 0) { ssize_t rrp = pread(fd_p, par_p, stripe_size, stripe_offset); (void)rrp; }
    if (fd_q >= 0) { ssize_t rrq = pread(fd_q, par_q, stripe_size, stripe_offset); (void)rrq; }

    // 3) 计算 diff 并 XOR 到 P/Q（按包偏移）
    for (int d = 0; d < config.k && d < 16; d++) {
        for (int p = 0; p < config.w && p < 16; p++) {
            if (!final_new[d][p]) continue;
            unsigned char *base = (unsigned char*)meta->data_blocks[d][p].base_data;
            size_t offb = (size_t)p * pkt_sz;
            // diff = new ^ base (若无base，直接把new视为diff)
            if (base) {
                // 就地把 final_new 用作 diff，先 XOR base
                xor_update_simd((char*)final_new[d][p], (char*)base, pkt_sz);
                xor_update_simd(par_p + offb, (char*)final_new[d][p], pkt_sz);
                xor_update_simd(par_q + offb, (char*)final_new[d][p], pkt_sz);
            } else {
                xor_update_simd(par_p + offb, (char*)final_new[d][p], pkt_sz);
                xor_update_simd(par_q + offb, (char*)final_new[d][p], pkt_sz);
            }
        }
    }

    // 4) 写回 P/Q
    if (fd_p >= 0) { ssize_t wwp = pwrite(fd_p, par_p, stripe_size, stripe_offset); (void)wwp; }
    if (fd_q >= 0) { ssize_t wwq = pwrite(fd_q, par_q, stripe_size, stripe_offset); (void)wwq; }

    pool_free(global_pool, par_p);
    pool_free(global_pool, par_q);

cleanup_only:
    // 5) 清理推测状态与日志
    for (int d = 0; d < 16; d++) {
        for (int p = 0; p < 16; p++) {
            if (meta->data_blocks[d][p].is_speculative) {
                meta->data_blocks[d][p].is_speculative = 0;
                meta->data_blocks[d][p].write_count = 0;
            }
            if (meta->data_blocks[d][p].base_data) {
                free(meta->data_blocks[d][p].base_data);
                meta->data_blocks[d][p].base_data = NULL;
            }
            if (final_new[d][p]) { free(final_new[d][p]); final_new[d][p] = NULL; }
        }
    }
    meta->parix_offset = 0;
    meta->parix_count = 0;

    if (PARIX_DEBUG_LOG) {
        printf("PARIX重放条带%d: 重放完成\n", stripe_id);
    }
    return 0;
}

// 增强版PARIX推测性写入 - 支持地址分配协同和智能推测
static __attribute__((unused)) int parix_speculative_write_enhanced(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id,
                                           void *new_data, uint32_t data_size, BlockPos *allocation_context) {
    if (!g_parix_manager.parix_enabled || stripe_id >= (uint32_t)g_parix_manager.total_stripes) {
        return -1;
    }

// 边界检查：防止数组越界
    if (disk_id >= 16 || packet_id >= 16) {
        return -1;
    }
    
    stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[stripe_id];
    pthread_mutex_lock(&meta->parix_lock);
    int wrote_entry = 0;
    
    // 智能推测决策 - 结合地址分配策略
    int should_speculate = parix_smart_speculation(stripe_id, disk_id, packet_id, allocation_context);
    if (!should_speculate) {
        pthread_mutex_unlock(&meta->parix_lock);
        return 0; // 不进行推测性写入
    }

// 更新地址分配元数据
    if (allocation_context) {
        parix_update_allocation_meta(stripe_id, allocation_context, 1);
    }

// 构建增强版PARIX日志条目
    parix_log_entry_t entry = {
        .magic = 0x50415249, // 'PARI'
        .stripe_id = stripe_id,
        .disk_id = disk_id,
        .packet_id = packet_id,
        .data_size = data_size,
        .sequence = meta->parix_sequence++,
        .checksum = 0,
        .is_initial = !meta->data_blocks[disk_id][packet_id].is_speculative
    };
    
    // 计算校验和
    entry.checksum = crc32(0, (const unsigned char*)new_data, data_size);
    
    uint32_t entry_total_size = sizeof(parix_log_entry_t) + data_size;
    
    // 智能空间管理
    if (meta->parix_offset + entry_total_size > meta->parix_size) {
        // 分析当前推测状态，决定是否重放
        int replay_benefit = parix_predict_speculation_benefit(stripe_id, disk_id, packet_id);
        if (replay_benefit > 0) {
            pthread_mutex_unlock(&meta->parix_lock);
            parix_replay_stripe(stripe_id);
            pthread_mutex_lock(&meta->parix_lock);
            meta->parix_offset = 0; // 重置偏移
        } else {
            // 扩展PARIX空间
            meta->parix_size *= 2;
            pthread_mutex_unlock(&meta->parix_lock);
            return -1; // 空间不足
        }
    }

// 批量写入优化
    struct iovec iov[2];
    iov[0].iov_base = &entry;
    iov[0].iov_len = sizeof(entry);
    iov[1].iov_base = new_data;
    iov[1].iov_len = data_size;
    
    off_t write_offset = g_parix_manager.parix_base_offset + 
                        (off_t)stripe_id * meta->parix_size + meta->parix_offset;
    
    // 使用writev进行批量写入
    if (g_parix_ring_ready && g_bufpool.enabled && g_fixed.enabled && g_parix_manager.parix_fd >= 0) {
        void *ptr = NULL; int idx = bufpool_acquire(&ptr);
        if (idx >= 0 && (size_t)g_bufpool.iovecs[idx].iov_len >= entry_total_size) {
            memcpy(ptr, &entry, sizeof(entry));
            memcpy((char*)ptr + sizeof(entry), new_data, data_size);
            struct io_uring_sqe *sqe = io_uring_get_sqe(&g_parix_ring);
            if (!sqe) { bufpool_release(idx); pthread_mutex_unlock(&meta->parix_lock); return -1; }
            io_uring_prep_write_fixed(sqe, g_parix_manager.parix_fd, ptr, (unsigned)entry_total_size, write_offset, idx);
            if (g_fixed.enabled) io_uring_sqe_set_flags(sqe, IOSQE_FIXED_FILE);
            if (io_uring_submit(&g_parix_ring) < 0) { bufpool_release(idx); pthread_mutex_unlock(&meta->parix_lock); return -1; }
            struct io_uring_cqe *cqe; if (io_uring_wait_cqe(&g_parix_ring, &cqe) < 0) { bufpool_release(idx); pthread_mutex_unlock(&meta->parix_lock); return -1; }
            int ok = (cqe->res == (int)entry_total_size);
            io_uring_cqe_seen(&g_parix_ring, cqe);
            bufpool_release(idx);
            if (!ok) { pthread_mutex_unlock(&meta->parix_lock); return -1; }
        } else {
            if (idx >= 0) bufpool_release(idx);
            ssize_t written = pwritev(g_parix_manager.parix_fd, iov, 2, write_offset);
            if (written != (ssize_t)entry_total_size) { pthread_mutex_unlock(&meta->parix_lock); return -1; }
        }
    } else {
        ssize_t written = pwritev(g_parix_manager.parix_fd, iov, 2, write_offset);
        if (written != (ssize_t)entry_total_size) { pthread_mutex_unlock(&meta->parix_lock); return -1; }
    }
    wrote_entry = 1;

// 更新推测状态和统计
    meta->data_blocks[disk_id][packet_id].is_speculative = 1;
    meta->data_blocks[disk_id][packet_id].write_count++;
    clock_gettime(CLOCK_MONOTONIC, &meta->data_blocks[disk_id][packet_id].last_write);
    
    // 更新地址分配提示
    int strategy = meta->allocation_meta.current_allocation_strategy;
    meta->data_blocks[disk_id][packet_id].allocation_hint = strategy;
    
    meta->parix_offset += entry_total_size;
    meta->parix_count++;
    
    // 更新全局统计
    pthread_mutex_lock(&stats.lock);
    stats.xor_count++;
    pthread_mutex_unlock(&stats.lock);
    
    pthread_mutex_unlock(&meta->parix_lock);
    
    printf("PARIX增强推测性写入: 条带%d, 磁盘%d, 包%d, 大小%d, 序列%d, %s\n",
           stripe_id, disk_id, packet_id, data_size, entry.sequence,
           entry.is_initial ? "初始" : "推测");
    
    return wrote_entry;
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

// 增强版性能监控函数
static __attribute__((unused)) void update_detailed_performance_stats(double io_time, int operation_count, int is_write) {
    pthread_mutex_lock(&stats.lock);
    
    // 更新基础统计
    if (is_write) {
        stats.write_count += operation_count;
    } else {
        stats.read_count += operation_count;
    }
    
    stats.total_io_time += io_time;
    stats.update_count++;
    
    // 计算详细指标
    if (io_time > 0) {
        double throughput = (operation_count * config.packetsize) / (io_time * 1024.0 * 1024.0); // MB/s
        stats.detailed_stats.throughput_mbps = throughput;
        stats.detailed_stats.iops = operation_count / io_time;
        stats.detailed_stats.avg_latency_ms = (io_time / operation_count) * 1000.0;
    }

// 更新实时统计
    stats.realtime_stats.current_throughput = stats.detailed_stats.throughput_mbps;
    stats.realtime_stats.current_latency = stats.detailed_stats.avg_latency_ms;
    stats.realtime_stats.active_operations = operation_count;
    
    // 更新历史数据
    if (stats.history.throughput_history && stats.history.latency_history) {
        stats.history.throughput_history[stats.history.history_index] = stats.detailed_stats.throughput_mbps;
        stats.history.latency_history[stats.history.history_index] = stats.detailed_stats.avg_latency_ms;
        stats.history.history_index = (stats.history.history_index + 1) % stats.history.history_size;
    }
    
    pthread_mutex_unlock(&stats.lock);
}

// 性能报告生成函数
static __attribute__((unused)) void generate_performance_report(void) {
    pthread_mutex_lock(&stats.lock);
    
    printf("\n=== 增强版性能报告 ===\n");
    printf("基础统计:\n");
    printf("  总I/O时间: %.3f秒\n", stats.total_io_time);
    printf("  计算时间: %.3f秒\n", stats.compute_time);
    printf("  读取次数: %d\n", stats.read_count);
    printf("  写入次数: %d\n", stats.write_count);
    printf("  更新次数: %d\n", stats.update_count);
    printf("  XOR操作次数: %lld\n", stats.xor_count);
    
    printf("\n详细性能指标:\n");
    printf("  吞吐量: %.2f MB/s\n", stats.detailed_stats.throughput_mbps);
    printf("  IOPS: %.2f\n", stats.detailed_stats.iops);
    printf("  平均延迟: %.3f ms\n", stats.detailed_stats.avg_latency_ms);
    printf("  95%%延迟: %.3f ms\n", stats.detailed_stats.p95_latency_ms);
    printf("  99%%延迟: %.3f ms\n", stats.detailed_stats.p99_latency_ms);
    printf("  CPU使用率: %.1f%%\n", stats.detailed_stats.cpu_usage_percent);
    printf("  内存使用: %.2f MB\n", stats.detailed_stats.memory_usage_mb);
    printf("  缓存命中: %lld\n", stats.detailed_stats.cache_hits);
    printf("  缓存未命中: %lld\n", stats.detailed_stats.cache_misses);
    printf("  SIMD效率: %.2f%%\n", stats.detailed_stats.simd_efficiency);
    printf("  I/O效率: %.2f%%\n", stats.detailed_stats.io_efficiency);
    printf("  校验效率: %.2f%%\n", stats.detailed_stats.parity_efficiency);
    
    printf("\n实时监控:\n");
    printf("  当前吞吐量: %.2f MB/s\n", stats.realtime_stats.current_throughput);
    printf("  当前延迟: %.3f ms\n", stats.realtime_stats.current_latency);
    printf("  活跃操作: %d\n", stats.realtime_stats.active_operations);
    printf("  队列深度: %d\n", stats.realtime_stats.queue_depth);
    printf("  负载因子: %.2f\n", stats.realtime_stats.load_factor);
    
    pthread_mutex_unlock(&stats.lock);
}

// 性能历史数据分析
static __attribute__((unused)) void analyze_performance_history(void) {
    if (!stats.history.throughput_history || !stats.history.latency_history) {
        return;
    }
    
    pthread_mutex_lock(&stats.lock);
    
    // 计算吞吐量趋势
    double avg_throughput = 0.0;
    double max_throughput = 0.0;
    double min_throughput = 1e10;
    
    for (int i = 0; i < stats.history.history_size; i++) {
        if (stats.history.throughput_history[i] > 0) {
            avg_throughput += stats.history.throughput_history[i];
            if (stats.history.throughput_history[i] > max_throughput) {
                max_throughput = stats.history.throughput_history[i];
            }
            if (stats.history.throughput_history[i] < min_throughput) {
                min_throughput = stats.history.throughput_history[i];
            }
        }
    }
    
    avg_throughput /= stats.history.history_size;
    
    printf("\n=== 性能历史分析 ===\n");
    printf("吞吐量分析:\n");
    printf("  平均吞吐量: %.2f MB/s\n", avg_throughput);
    printf("  最大吞吐量: %.2f MB/s\n", max_throughput);
    printf("  最小吞吐量: %.2f MB/s\n", min_throughput);
    printf("  吞吐量波动: %.2f%%\n", ((max_throughput - min_throughput) / avg_throughput) * 100.0);
    
    pthread_mutex_unlock(&stats.lock);
}

// 初始化增强版性能监控
static __attribute__((unused)) void init_enhanced_performance_monitoring(int history_size) {
    pthread_mutex_lock(&stats.lock);
    
    // 初始化历史数据
    stats.history.history_size = history_size;
    stats.history.history_index = 0;
    stats.history.throughput_history = malloc(history_size * sizeof(double));
    stats.history.latency_history = malloc(history_size * sizeof(double));
    
    if (stats.history.throughput_history && stats.history.latency_history) {
        memset(stats.history.throughput_history, 0, history_size * sizeof(double));
        memset(stats.history.latency_history, 0, history_size * sizeof(double));
    }

// 初始化详细统计
    memset(&stats.detailed_stats, 0, sizeof(stats.detailed_stats));
    memset(&stats.realtime_stats, 0, sizeof(stats.realtime_stats));
    
    pthread_mutex_unlock(&stats.lock);
    
    printf("增强版性能监控已初始化，历史数据大小: %d\n", history_size);
}

// 清理增强版性能监控
static __attribute__((unused)) void cleanup_enhanced_performance_monitoring(void) {
    pthread_mutex_lock(&stats.lock);
    
    if (stats.history.throughput_history) {
        free(stats.history.throughput_history);
        stats.history.throughput_history = NULL;
    }
    
    if (stats.history.latency_history) {
        free(stats.history.latency_history);
        stats.history.latency_history = NULL;
    }
    
    pthread_mutex_unlock(&stats.lock);
}

// 增强版错误处理系统
typedef struct {
    int error_count;
    int warning_count;
    int fatal_count;
    char last_error[256];
    char last_warning[256];
    struct timespec last_error_time;
    pthread_mutex_t error_lock;
} error_handler_t;

static error_handler_t g_error_handler = {0};

// 错误级别定义
typedef enum {
    ERROR_LEVEL_INFO = 0,
    ERROR_LEVEL_WARNING = 1,
    ERROR_LEVEL_ERROR = 2,
    ERROR_LEVEL_FATAL = 3
} error_level_t;

// 增强版错误记录函数
static __attribute__((unused)) void log_error_enhanced(error_level_t level, const char *function, int line, const char *format, ...) {
    va_list args;
    va_start(args, format);
    
    pthread_mutex_lock(&g_error_handler.error_lock);
    
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    
    const char *level_str[] = {"INFO", "WARNING", "ERROR", "FATAL"};
    const char *level_color[] = {"\033[32m", "\033[33m", "\033[31m", "\033[35m"};
    const char *reset_color = "\033[0m";
    
    // 格式化错误消息
    char error_msg[512];
    vsnprintf(error_msg, sizeof(error_msg), format, args);
    
    // 输出带颜色的错误信息
    printf("%s[%s] %s:%d - %s%s\n", 
           level_color[level], level_str[level], function, line, error_msg, reset_color);
    
    // 更新错误统计
    switch (level) {
        case ERROR_LEVEL_WARNING:
            g_error_handler.warning_count++;
            strncpy(g_error_handler.last_warning, error_msg, sizeof(g_error_handler.last_warning) - 1);
            break;
        case ERROR_LEVEL_ERROR:
            g_error_handler.error_count++;
            strncpy(g_error_handler.last_error, error_msg, sizeof(g_error_handler.last_error) - 1);
            break;
        case ERROR_LEVEL_FATAL:
            g_error_handler.fatal_count++;
            strncpy(g_error_handler.last_error, error_msg, sizeof(g_error_handler.last_error) - 1);
            break;
        default:
            break;
    }
    
    g_error_handler.last_error_time = now;
    
    pthread_mutex_unlock(&g_error_handler.error_lock);
    
    va_end(args);
    
    // 致命错误时退出
    if (level == ERROR_LEVEL_FATAL) {
        printf("致命错误，程序退出\n");
        exit(1);
    }
}

// 初始化错误处理系统
static __attribute__((unused)) void init_error_handling_system(void) {
    pthread_mutex_init(&g_error_handler.error_lock, NULL);
    memset(&g_error_handler, 0, sizeof(g_error_handler));
    
    printf("增强版错误处理系统已初始化\n");
}

// 清理错误处理系统
static __attribute__((unused)) void cleanup_error_handling_system(void) {
    pthread_mutex_destroy(&g_error_handler.error_lock);
}

static int demo_init_rs(int total_stripes) {
    if (total_stripes <= 0) total_stripes = 1;
    if (g_rs_manager.rs_enabled) return 0;
    if (rs_manager_init_optimized(total_stripes, 0) != 0) {
        fprintf(stderr, "RS demo init failed; Reserved Space logging disabled for this run.\n");
        return -1;
    }
    return 0;
}

static void demo_shutdown_rs(void) {
    if (g_rs_manager.rs_enabled) {
        rs_manager_destroy_optimized();
    }
}

static int demo_init_parix(int total_stripes) {
    const size_t per_stripe = 256 * 1024;
    if (total_stripes <= 0) total_stripes = 1;

    memset(&g_parix_manager, 0, sizeof(g_parix_manager));
    g_parix_manager.total_stripes = total_stripes;
    g_parix_manager.parix_enabled = 1;
    g_parix_manager.parix_base_offset = 0;
    g_parix_manager.replay_thread_running = 0;
    snprintf(g_parix_manager.parix_file_path,
             sizeof(g_parix_manager.parix_file_path),
             "/tmp/evenodd_parix_opt_%d.dat", getpid());

    off_t file_size = (off_t)total_stripes * (off_t)per_stripe;
    fprintf(stderr, "[FD-CHECK] PARIX: opening file %s\n", g_parix_manager.parix_file_path);
    g_parix_manager.parix_fd =
        open(g_parix_manager.parix_file_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (g_parix_manager.parix_fd < 0) {
        fprintf(stderr, "PARIX demo init failed: %s\n", strerror(errno));
        memset(&g_parix_manager, 0, sizeof(g_parix_manager));
        return -1;
    }
    fprintf(stderr, "[FD-CHECK] PARIX: opened with fd=%d\n", g_parix_manager.parix_fd);
    
    // 确保 fd >= 3，避免与 stdio (0/1/2) 冲突
    g_parix_manager.parix_fd = ensure_nonstdio_fd(g_parix_manager.parix_fd, "PARIX");
    fprintf(stderr, "[FD-CHECK] PARIX: final fd=%d\n", g_parix_manager.parix_fd);
    
    if (ftruncate(g_parix_manager.parix_fd, file_size) != 0) {
        fprintf(stderr, "PARIX demo truncate failed: %s\n", strerror(errno));
        close(g_parix_manager.parix_fd);
        unlink(g_parix_manager.parix_file_path);
        memset(&g_parix_manager, 0, sizeof(g_parix_manager));
        return -1;
    }

    g_parix_manager.stripe_metas =
        (stripe_parix_meta_t*)calloc((size_t)total_stripes, sizeof(stripe_parix_meta_t));
    if (!g_parix_manager.stripe_metas) {
        fprintf(stderr, "PARIX demo: unable to allocate stripe metadata\n");
        close(g_parix_manager.parix_fd);
        unlink(g_parix_manager.parix_file_path);
        memset(&g_parix_manager, 0, sizeof(g_parix_manager));
        return -1;
    }

    for (int i = 0; i < total_stripes; i++) {
        stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[i];
        meta->parix_offset = 0;
        meta->parix_size = (uint32_t)per_stripe;
        meta->parix_sequence = 1;
        meta->parix_count = 0;
        meta->allocation_meta.current_allocation_strategy = 0;
        pthread_mutex_init(&meta->parix_lock, NULL);
    }
    pthread_mutex_init(&g_parix_manager.replay_mu, NULL);
    pthread_cond_init(&g_parix_manager.replay_cv, NULL);
    g_parix_manager.need_replay = 0;
    
    // 初始化 PARIX io_uring 环境（用于前台 write_fixed）
    if (!g_parix_ring_ready && g_fixed.enabled) {
        if (io_uring_queue_init(256, &g_parix_ring, 0) == 0) {
            g_parix_ring_ready = 1;
            // 若固定缓冲池未启用，则尝试按单条目最大长度注册缓冲
            if (!g_bufpool.enabled) {
                size_t buf_size = sizeof(parix_log_entry_t) + (size_t)config.packetsize;
                (void)bufpool_init(&g_parix_ring, 512, buf_size);
            }
        }
    }
    return 0;
}

static void demo_shutdown_parix(void) {
    if (g_parix_manager.stripe_metas) {
        for (int i = 0; i < g_parix_manager.total_stripes; i++) {
            stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[i];
            for (int d = 0; d < 16; d++) {
                for (int p = 0; p < 16; p++) {
                    if (meta->data_blocks[d][p].base_data) {
                        free(meta->data_blocks[d][p].base_data);
                        meta->data_blocks[d][p].base_data = NULL;
                    }
                }
            }
            pthread_mutex_destroy(&meta->parix_lock);
        }
        free(g_parix_manager.stripe_metas);
    }
    if (g_parix_manager.parix_fd >= 0) {
        close(g_parix_manager.parix_fd);
        if (g_parix_manager.parix_file_path[0] != '\0') {
            unlink(g_parix_manager.parix_file_path);
        }
    }
    pthread_mutex_destroy(&g_parix_manager.replay_mu);
    pthread_cond_destroy(&g_parix_manager.replay_cv);
    memset(&g_parix_manager, 0, sizeof(g_parix_manager));
    if (g_parix_ring_ready) {
        io_uring_queue_exit(&g_parix_ring);
        g_parix_ring_ready = 0;
    }
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
    fprintf(stderr, "  -R            Enable Reserved Space logging paths\n");
    fprintf(stderr, "  -S            Enable strong consistency revalidation flag\n");
    fprintf(stderr, "  -P            Enable PARIX speculative write hints\n");
    fprintf(stderr, "  -t <count>    PARIX speculation threshold\n");
    fprintf(stderr, "  -V <count>    Enable verification (sample stripes, <=0 for full)\n");
    fprintf(stderr, "  -h            Show this help and exit\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "If an input_file is provided its first stripe is encoded; otherwise synthetic data is used.\n");
}

int main(int argc, char *argv[]) {
    int opt;
    const char *input_path = NULL;
    int exit_code = 0;
    int stats_lock_initialized = 0;
    int input_fd = -1;
    int input_exhausted = 0;
    int rs_ready = 0;
    int parix_ready = 0;
    const int demo_stripes = 4;
    int w_option_explicit = 0;

    while ((opt = getopt(argc, argv, "k:m:w:p:u:n:a:RSPt:V:h")) != -1) {
        switch (opt) {
            case 'k': config.k = atoi(optarg); break;
            case 'm': config.m = atoi(optarg); break;
            case 'w': config.w = atoi(optarg); w_option_explicit = 1; break;
            case 'p': config.packetsize = atoi(optarg); break;
            case 'u': config.update_size = atoi(optarg); break;
            case 'n': config.n_updates = atoi(optarg); break;
            case 'a': config.alloc = optarg; break;
            case 'R': config.use_rs = 1; break;
            case 'S': config.strong = 1; break;
            case 'P': config.use_parix = 1; break;
            case 't': config.parix_threshold = atoi(optarg); break;
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
    printf("Reserved Space logging: %s\n", config.use_rs ? "enabled" : "disabled");
    printf("PARIX speculation: %s\n", config.use_parix ? "enabled" : "disabled");

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

    if (config.use_rs) {
        if (demo_init_rs(demo_stripes) == 0) {
            rs_ready = 1;
            void *delta = malloc((size_t)config.packetsize);
            if (!delta) {
                fprintf(stderr, "RS demo: unable to allocate delta buffer\n");
            } else {
                memcpy(delta, coding[0], (size_t)config.packetsize);
                if (rs_log_write(0, 0, 0, 0, delta, (uint32_t)config.packetsize) != 0) {
                    fprintf(stderr, "RS demo: failed to append P-delta entry\n");
                }
                memset(delta, 0x5a, (size_t)config.packetsize);
                if (rs_log_write(0, 1 % config.k, 0, 1, delta, (uint32_t)config.packetsize) != 0) {
                    fprintf(stderr, "RS demo: failed to append Q-delta entry\n");
                }
                free(delta);
            }
            if (g_rs_manager.stripe_metas) {
                stripe_rs_meta_t *meta = &g_rs_manager.stripe_metas[0];
                printf("\nRS demo summary:\n");
                printf("  Stripe 0 -> entries=%llu, P_offset=%llu bytes, Q_offset=%llu bytes\n",
                       (unsigned long long)meta->log_count,
                       (unsigned long long)meta->p_rs_offset,
                       (unsigned long long)meta->q_rs_offset);
            }
        } else {
            config.use_rs = 0;
        }
    }

    if (config.use_parix) {
        if (demo_init_parix(demo_stripes) == 0) {
            parix_ready = 1;
            void *spec_buf = malloc((size_t)config.packetsize);
            if (!spec_buf) {
                fprintf(stderr, "PARIX demo: unable to allocate speculative buffer\n");
            } else {
                memset(spec_buf, 0x2a, (size_t)config.packetsize);
                if (parix_speculative_write(0, 0, 0, spec_buf,
                                            (uint32_t)config.packetsize) != 0) {
                    fprintf(stderr, "PARIX demo: first speculative write failed\n");
                }
                memset(spec_buf, 0x6c, (size_t)config.packetsize);
                if (parix_speculative_write(0, 0, 0, spec_buf,
                                            (uint32_t)config.packetsize) != 0) {
                    fprintf(stderr, "PARIX demo: second speculative write failed\n");
                }
                free(spec_buf);
            }
            if (g_parix_manager.stripe_metas) {
                stripe_parix_meta_t *meta = &g_parix_manager.stripe_metas[0];
                printf("\nPARIX demo summary (before replay):\n");
                printf("  Stripe 0 -> entries=%u, offset=%u bytes, next_seq=%u\n",
                       meta->parix_count, meta->parix_offset, meta->parix_sequence);
                parix_replay_stripe(0);
                printf("  After replay: entries=%u, offset=%u\n",
                       meta->parix_count, meta->parix_offset);
                printf("  PARIX log file: %s\n", g_parix_manager.parix_file_path);
            }
        } else {
            config.use_parix = 0;
        }
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
    if (parix_ready) {
        demo_shutdown_parix();
    }
    if (rs_ready) {
        demo_shutdown_rs();
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
