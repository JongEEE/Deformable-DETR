#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# 该脚本启动整个训练，可以支持分布式训练，通过参数设置，会有再下一层的脚本进行分布式训练的配置
# 启动 launch.py，在该脚本里面，判断如果是分布式训练，会继续配置，以子进程的形式，开启训练

# 启动调试命令
set -x

# 命令行参数赋值
GPUS=$1
# 从第二个参数起的都会赋值给该变量
RUN_COMMAND=${@:2}
# 条件判断，进行变量赋值
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
# 变量赋值
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python ./tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}