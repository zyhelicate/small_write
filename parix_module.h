#ifndef PARIX_MODULE_H
#define PARIX_MODULE_H

#include <stddef.h>
#include <stdint.h>

#include "alloc_strategy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PARIX_MODE_BASIC = 0,
    PARIX_MODE_WITH_ALLOC = 1
} parix_mode_t;

typedef struct parix_local_ctx parix_local_ctx_t;

parix_local_ctx_t *parix_local_init(const char *base_dir,
                                    int k,
                                    int w,
                                    size_t packet_size,
                                    parix_mode_t mode);

int parix_local_submit(parix_local_ctx_t            *ctx,
                       uint32_t                     stripe_id,
                       const BlockPos              *plan,
                       int                          plan_count,
                       const unsigned char         *payload,
                       size_t                       packet_size);

int parix_local_replay(parix_local_ctx_t *ctx);

void parix_local_shutdown(parix_local_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PARIX_MODULE_H */
