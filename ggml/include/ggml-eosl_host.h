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

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_eosl_host_init(int device);

GGML_BACKEND_API bool ggml_backend_is_eosl_host(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_eosl_host_buffer_type(int device);

GGML_BACKEND_API int ggml_backend_eosl_host_get_device_count(void);
GGML_BACKEND_API void ggml_backend_eosl_host_get_device_memory(const int device, size_t *free, size_t *total);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_eosl_host_reg(void);

#ifdef  __cplusplus
}
#endif
