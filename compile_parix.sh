#!/bin/bash

# EVENODD 编译脚本
# 编译地址分配策略模块化的版本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 编译参数
CC=gcc
CFLAGS="-O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -march=native"
LDFLAGS="-luring -lpthread -lm"
TARGET="evenodd_parix"
SRC_FILES="evenodd_parix.c alloc_strategy.c parix_module.c"

echo -e "${GREEN}开始编译 EVENODD...${NC}"

# 检查源文件是否存在
for file in $SRC_FILES; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}错误: 源文件 $file 不存在${NC}"
        exit 1
    fi
done

# 检查 liburing 是否安装
if ! pkg-config --exists liburing 2>/dev/null; then
    echo -e "${YELLOW}警告: 未检测到 liburing pkg-config，尝试直接编译...${NC}"
fi

# 编译
echo -e "${GREEN}编译命令:${NC}"
echo "$CC $CFLAGS -o $TARGET $SRC_FILES $LDFLAGS"
echo ""

$CC $CFLAGS -o $TARGET $SRC_FILES $LDFLAGS

if [ $? -eq 0 ]; then
    echo -e "${GREEN}编译成功！可执行文件: $TARGET${NC}"
    ls -lh $TARGET
else
    echo -e "${RED}编译失败！${NC}"
    exit 1
fi

