#include "alloc_strategy.h"

#include <string.h>

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

int map_blocks_row(int s, int k, int w, BlockPos *blocks) {
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

int map_blocks_diag(int s, int k, int w, BlockPos *blocks) {
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

int choose_best_mapping(int s, int k, int w, BlockPos *blocks) {
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

int map_blocks_row_optimized(int s, int k, int w, BlockPos *blocks) {
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

int map_blocks_diag_fixed(int s, int k, int w, int target_diag, BlockPos *blocks) {
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

static int calculate_mapping_cost_penalized(BlockPos *blocks, int s, int k, int w) {
    int base = calculate_mapping_cost(blocks, s, k, w);
    for (int i = 0; i < s; i++) {
        if (blocks[i].diag % (k - 1) == 0) { base += (k - 2); break; }
    }
    return base;
}

int map_blocks_diag_multi_nonzero(int s, int k, int w, BlockPos *blocks) {
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

int choose_best_mapping_enhanced(int s, int k, int w, BlockPos *blocks) {
    BlockPos row_opt[s];
    BlockPos diag_opt[s];
    map_blocks_row_optimized(s, k, w, row_opt);
    map_blocks_diag_fixed(s, k, w, 1, diag_opt);
    int c_row = calculate_mapping_cost_penalized(row_opt, s, k, w);
    int c_diag = calculate_mapping_cost_penalized(diag_opt, s, k, w);
    if (c_row <= c_diag) { memcpy(blocks, row_opt, (size_t)s * sizeof(BlockPos)); return s; }
    memcpy(blocks, diag_opt, (size_t)s * sizeof(BlockPos)); return s;
}

int map_blocks_rs_friendly(int s, int k, int w, BlockPos *blocks) {
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

int map_blocks_batch_aware(int s, int k, int w, int batch_size, BlockPos *blocks) {
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

int map_blocks_rs_load_aware(int s, int k, int w, double current_rs_load, BlockPos *blocks) {
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

int map_blocks_rs_ultra_compact(int s, int k, int w, int max_stripes_per_batch, BlockPos *blocks) {
    if (s <= 0) return 0;
    int blocks_per_stripe = k * w;
    int stripes_needed = (s + blocks_per_stripe - 1) / blocks_per_stripe;
    if (stripes_needed > max_stripes_per_batch) stripes_needed = max_stripes_per_batch;
    int placed = 0;
    int current_stripe = 0;
    while (placed < s) {
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

int map_blocks_stripe_hotspot(int s, int k, int w, int hotspot_stripe_id, BlockPos *blocks) {
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
