#include "parix_module.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define PARIX_FLAG_HAS_ORIGINAL 0x1u

typedef struct {
    uint32_t stripe_id;
    uint16_t disk_id;
    uint16_t packet_id;
    uint16_t row_id;
    uint16_t flags;
    unsigned char *original;
    unsigned char *latest;
    size_t length;
    int has_original;
    int dirty;
} parix_block_state_t;

struct parix_local_ctx {
    parix_mode_t mode;
    int k;
    int w;
    size_t packet_size;
    unsigned long long sequence;
    char base_dir[PATH_MAX];
    char log_path[PATH_MAX];
    FILE *log_fp;

    parix_block_state_t *blocks;
    size_t block_count;
    size_t block_capacity;

    unsigned char *parity_buffer;
    size_t parity_size;

    /* 快速索引：开放寻址哈希表 (key -> block index+1) */
    size_t map_size;           /* 表大小（2 的幂） */
    uint64_t *map_keys;        /* 0 表示空槽 */
    uint32_t *map_vals;        /* 存 block 索引+1，0 表示空 */
};

static inline uint64_t parix_key(uint32_t stripe_id, uint16_t disk_id, uint16_t packet_id) {
    return (((uint64_t)stripe_id) << 32) | (((uint64_t)disk_id) << 16) | (uint64_t)packet_id;
}

static int map_init(parix_local_ctx_t *ctx, size_t initial) {
    size_t n = 1;
    while (n < initial) n <<= 1;
    ctx->map_size = n ? n : 64;
    ctx->map_keys = (uint64_t*)calloc(ctx->map_size, sizeof(uint64_t));
    ctx->map_vals = (uint32_t*)calloc(ctx->map_size, sizeof(uint32_t));
    return (ctx->map_keys && ctx->map_vals) ? 0 : -1;
}

static void map_destroy(parix_local_ctx_t *ctx) {
    free(ctx->map_keys); ctx->map_keys = NULL;
    free(ctx->map_vals); ctx->map_vals = NULL;
    ctx->map_size = 0;
}

static int map_put(parix_local_ctx_t *ctx, uint64_t key, uint32_t val_plus1) {
    size_t mask = ctx->map_size - 1;
    size_t i = (size_t)(key * 11400714819323198485ull) & mask;
    for (size_t p = 0; p < ctx->map_size; p++) {
        size_t idx = (i + p) & mask;
        if (ctx->map_keys[idx] == 0 || ctx->map_keys[idx] == key) {
            ctx->map_keys[idx] = key;
            ctx->map_vals[idx] = val_plus1;
            return 0;
        }
    }
    return -1;
}

static uint32_t map_get(parix_local_ctx_t *ctx, uint64_t key) {
    size_t mask = ctx->map_size - 1;
    size_t i = (size_t)(key * 11400714819323198485ull) & mask;
    for (size_t p = 0; p < ctx->map_size; p++) {
        size_t idx = (i + p) & mask;
        uint64_t k = ctx->map_keys[idx];
        if (k == 0) return 0;
        if (k == key) return ctx->map_vals[idx];
    }
    return 0;
}

static int ensure_directory(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return 0;
        }
        errno = ENOTDIR;
        return -1;
    }
    if (mkdir(path, 0755) == 0) {
        return 0;
    }
    return -1;
}

static parix_block_state_t *find_block(parix_local_ctx_t *ctx,
                                       uint32_t stripe_id,
                                       uint16_t disk_id,
                                       uint16_t packet_id) {
    if (ctx->map_size == 0) return NULL;
    uint64_t key = parix_key(stripe_id, disk_id, packet_id);
    uint32_t v = map_get(ctx, key);
    if (v == 0) return NULL;
    uint32_t idx = v - 1;
    if (idx >= ctx->block_count) return NULL;
    return &ctx->blocks[idx];
}

static parix_block_state_t *alloc_block(parix_local_ctx_t *ctx,
                                        uint32_t stripe_id,
                                        uint16_t disk_id,
                                        uint16_t packet_id,
                                        uint16_t row_id) {
    if (ctx->block_count == ctx->block_capacity) {
        size_t new_cap = ctx->block_capacity ? ctx->block_capacity * 2 : 64;
        parix_block_state_t *tmp = (parix_block_state_t*)realloc(ctx->blocks, new_cap * sizeof(parix_block_state_t));
        if (!tmp) {
            return NULL;
        }
        ctx->blocks = tmp;
        ctx->block_capacity = new_cap;
    }
    parix_block_state_t *st = &ctx->blocks[ctx->block_count++];
    memset(st, 0, sizeof(*st));
    st->stripe_id = stripe_id;
    st->disk_id = disk_id;
    st->packet_id = packet_id;
    st->row_id = row_id;
    st->length = ctx->packet_size;
    st->original = (unsigned char*)calloc(ctx->packet_size, 1);
    st->latest = (unsigned char*)malloc(ctx->packet_size);
    if (!st->original || !st->latest) {
        free(st->original);
        free(st->latest);
        ctx->block_count--;
        return NULL;
    }
    memset(st->latest, 0, ctx->packet_size);
    /* 放入哈希索引 */
    uint64_t key = parix_key(stripe_id, disk_id, packet_id);
    if (ctx->map_size == 0) {
        map_init(ctx, 128);
    }
    (void)map_put(ctx, key, (uint32_t)ctx->block_count); /* block_count 已自增，正好是 idx+1 */
    return st;
}

parix_local_ctx_t *parix_local_init(const char *base_dir,
                                    int k,
                                    int w,
                                    size_t packet_size,
                                    parix_mode_t mode) {
    parix_local_ctx_t *ctx = (parix_local_ctx_t*)calloc(1, sizeof(parix_local_ctx_t));
    if (!ctx) return NULL;

    ctx->mode = mode;
    ctx->k = k;
    ctx->w = w;
    ctx->packet_size = packet_size;
    ctx->sequence = 0ULL;

    const char *dir = base_dir && base_dir[0] ? base_dir : "./parix_local";
    snprintf(ctx->base_dir, sizeof(ctx->base_dir), "%s", dir);

    if (ensure_directory(ctx->base_dir) != 0) {
        free(ctx);
        return NULL;
    }

    snprintf(ctx->log_path, sizeof(ctx->log_path), "%s/parix.log", ctx->base_dir);
    ctx->log_fp = fopen(ctx->log_path, "ab+");
    if (!ctx->log_fp) {
        free(ctx);
        return NULL;
    }

    ctx->parity_size = (size_t)w * packet_size;
    ctx->parity_buffer = (unsigned char*)calloc(ctx->parity_size, 1);
    if (!ctx->parity_buffer) {
        fclose(ctx->log_fp);
        free(ctx);
        return NULL;
    }

    ctx->blocks = NULL;
    ctx->block_count = 0;
    ctx->block_capacity = 0;

    /* 设置日志大缓冲，减少 fwrite 调用的同步成本 */
    setvbuf(ctx->log_fp, NULL, _IOFBF, 4 * 1024 * 1024);

    /* 初始化索引 */
    map_init(ctx, 128);

    return ctx;
}

typedef struct __attribute__((packed)) {
    uint32_t stripe_id;
    uint16_t disk_id;
    uint16_t packet_id;
    uint16_t row_id;
    uint16_t flags;
    uint32_t length;
    uint64_t sequence;
} parix_log_hdr_t;

static int append_log(parix_local_ctx_t *ctx,
                      const parix_log_hdr_t *hdr,
                      const unsigned char *original,
                      const unsigned char *latest) {
    if (!ctx->log_fp) return -1;
    if (fwrite(hdr, sizeof(*hdr), 1, ctx->log_fp) != 1) return -1;
    if (hdr->flags & PARIX_FLAG_HAS_ORIGINAL) {
        if (fwrite(original, hdr->length, 1, ctx->log_fp) != 1) return -1;
    }
    if (fwrite(latest, hdr->length, 1, ctx->log_fp) != 1) return -1;
    /* 不在每条记录上 fflush；交给 OS 缓冲，或在上层阶段性刷新 */
    return 0;
}

int parix_local_submit(parix_local_ctx_t            *ctx,
                       uint32_t                     stripe_id,
                       const BlockPos              *plan,
                       int                          plan_count,
                       const unsigned char         *payload,
                       size_t                       packet_size) {
    if (!ctx || !plan || plan_count <= 0 || !payload) return -1;
    size_t expected = ctx->packet_size;
    if (packet_size != expected) return -1;

    for (int i = 0; i < plan_count; i++) {
        const BlockPos *bp = &plan[i];
        uint16_t disk_id = (uint16_t)(bp->col >= 0 ? bp->col : 0);
        uint16_t packet_id = (uint16_t)(bp->diag >= 0 ? bp->diag : 0);
        uint16_t row_id = (uint16_t)(bp->row >= 0 ? bp->row : 0);
        const unsigned char *src = payload + (size_t)i * packet_size;

        parix_block_state_t *st = find_block(ctx, stripe_id, disk_id, packet_id);
        if (!st) {
            st = alloc_block(ctx, stripe_id, disk_id, packet_id, row_id);
            if (!st) {
                return -1;
            }
        }

        parix_log_hdr_t hdr;
        hdr.stripe_id = stripe_id;
        hdr.disk_id = disk_id;
        hdr.packet_id = packet_id;
        hdr.row_id = row_id;
        hdr.flags = 0;
        hdr.length = (uint32_t)packet_size;
        hdr.sequence = ++ctx->sequence;

        if (!st->has_original) {
            memcpy(st->original, st->latest, packet_size);
            hdr.flags |= PARIX_FLAG_HAS_ORIGINAL;
            st->has_original = 1;
        }

        memcpy(st->latest, src, packet_size);
        st->dirty = 1;

        if (append_log(ctx, &hdr,
                       (hdr.flags & PARIX_FLAG_HAS_ORIGINAL) ? st->original : NULL,
                       st->latest) != 0) {
            return -1;
        }
    }
    return 0;
}

int parix_local_replay(parix_local_ctx_t *ctx) {
    if (!ctx) return -1;
    if (!ctx->parity_buffer) return -1;

    memset(ctx->parity_buffer, 0, ctx->parity_size);

    for (size_t i = 0; i < ctx->block_count; i++) {
        parix_block_state_t *st = &ctx->blocks[i];
        if (!st->dirty) continue;
        size_t offset = ((size_t)(st->row_id % ctx->w)) * ctx->packet_size;
        if (offset + ctx->packet_size <= ctx->parity_size) {
            for (size_t b = 0; b < ctx->packet_size; b++) {
                unsigned char delta = st->latest[b] ^ st->original[b];
                ctx->parity_buffer[offset + b] ^= delta;
            }
        }
        st->dirty = 0;
    }

    return 0;
}

void parix_local_shutdown(parix_local_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->log_fp) {
        fflush(ctx->log_fp);
        fclose(ctx->log_fp);
    }
    if (ctx->parity_buffer) {
        char parity_path[PATH_MAX];
        snprintf(parity_path, sizeof(parity_path), "%s/parix_parity.bin", ctx->base_dir);
        FILE *pf = fopen(parity_path, "wb");
        if (pf) {
            fwrite(ctx->parity_buffer, ctx->parity_size, 1, pf);
            fclose(pf);
        }
        free(ctx->parity_buffer);
    }
    for (size_t i = 0; i < ctx->block_count; i++) {
        free(ctx->blocks[i].original);
        free(ctx->blocks[i].latest);
    }
    free(ctx->blocks);
    map_destroy(ctx);
    free(ctx);
}
