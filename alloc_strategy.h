#ifndef ALLOC_STRATEGY_H
#define ALLOC_STRATEGY_H

#include <stddef.h>

typedef struct {
    int row;   // 包索引 (0..w-1)
    int col;   // 磁盘索引 (0..k-1)
    int diag;  // 对角线索引 (0..k-2)
} BlockPos;

int map_blocks_row(int s, int k, int w, BlockPos *blocks);
int map_blocks_diag(int s, int k, int w, BlockPos *blocks);
int choose_best_mapping(int s, int k, int w, BlockPos *blocks);
int map_blocks_row_optimized(int s, int k, int w, BlockPos *blocks);
int map_blocks_diag_fixed(int s, int k, int w, int target_diag, BlockPos *blocks);
int map_blocks_diag_multi_nonzero(int s, int k, int w, BlockPos *blocks);
int choose_best_mapping_enhanced(int s, int k, int w, BlockPos *blocks);
int map_blocks_rs_friendly(int s, int k, int w, BlockPos *blocks);
int map_blocks_batch_aware(int s, int k, int w, int batch_size, BlockPos *blocks);
int map_blocks_rs_load_aware(int s, int k, int w, double current_rs_load, BlockPos *blocks);
int map_blocks_rs_ultra_compact(int s, int k, int w, int max_stripes_per_batch, BlockPos *blocks);
int map_blocks_stripe_hotspot(int s, int k, int w, int hotspot_stripe_id, BlockPos *blocks);

#endif /* ALLOC_STRATEGY_H */
