/*
 * Copyright (C) 2015-2019 EOSL/ITRI
 * All rights reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of EOSL/ITRI and its suppliers, if any.
 * The intellectual and technical concepts contained
 * herein are proprietary to EOSL/ITRI and its suppliers and
 * may be covered by Taiwan and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from EOSL/ITRI.
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_EOSL_NAME        "EOSL"
#define GGML_EOSL_MAX_DEVICES 16

// all EOSL device structures must be packed
#pragma pack(push, 1)

// ggml_tensor is serialized into c
struct __attribute__ ((packed, aligned(8))) ggml_eosl_serialized_tensor {
    uint64_t id;           // host side ggml_tensor address
    uint32_t type;
    uint64_t buffer_id;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    uint64_t data_size;
    char name[GGML_MAX_NAME];
};
typedef struct ggml_eosl_serialized_tensor *ggml_eosl_serialized_tensor_t;

static_assert(sizeof(ggml_eosl_serialized_tensor) % 8 == 0, "ggml_eosl_serialized_tensor size must be multiple of 8");

// EOSL device commands
enum ggml_eosl_dev_cmd {
    EOSL_GGML_DEV_CMD_ALLOC_BUFFER = 1, 
    EOSL_GGML_DEV_CMD_GET_ALIGNMENT,
    EOSL_GGML_DEV_CMD_GET_MAX_SIZE,
    EOSL_GGML_DEV_CMD_BUFFER_GET_BASE,
    EOSL_GGML_DEV_CMD_FREE_BUFFER,
    EOSL_GGML_DEV_CMD_BUFFER_CLEAR,
    EOSL_GGML_DEV_CMD_INIT_TENSOR,
    EOSL_GGML_DEV_CMD_SET_TENSOR,
    EOSL_GGML_DEV_CMD_GET_TENSOR,
    EOSL_GGML_DEV_CMD_MEMSET_TENSOR,
    EOSL_GGML_DEV_CMD_COPY_TENSOR,
    EOSL_GGML_DEV_CMD_GRAPH_COMPUTE,
    EOSL_GGML_DEV_CMD_GET_DEVICE_MEMORY,
    EOSL_GGML_DEV_CMD_COUNT,
};

#pragma pack(pop)

#ifdef  __cplusplus
}
#endif
