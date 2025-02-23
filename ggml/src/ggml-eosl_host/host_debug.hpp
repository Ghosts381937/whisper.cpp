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

#include "ggml-impl.h"

#define EOSL_LOG_ERROR(TAG, FMT, ...)                                          \
    GGML_LOG_ERROR("%s: [ERROR] " FMT "(%s:%d:%s)\n", TAG, ##__VA_ARGS__, __FILE__, __LINE__, __func__)

#define EOSL_LOG_ERROR_S(TAG, FMT, ...)                                        \
    GGML_LOG_ERROR("%s: [ERROR] " FMT "\n", TAG, ##__VA_ARGS__)

#define EOSL_LOG_WARN(TAG, FMT, ...)                                           \
    GGML_LOG_WARN("%s: [WARN] " FMT "(%s:%d:%s)\n", TAG, ##__VA_ARGS__, __FILE__, __LINE__, __func__)
    
#define EOSL_LOG_WARN_S(TAG, FMT, ...)                                         \
    GGML_LOG_WARN("%s: [WARN] " FMT "\n", TAG, ##__VA_ARGS__)

#define EOSL_LOG_INFO(TAG, FMT, ...)                                           \
    GGML_LOG_INFO("%s: [INFO] " FMT "\n", TAG, ##__VA_ARGS__)

#define EOSL_LOG_DEBUG(TAG, FMT, ...)                                          \
    GGML_LOG_DEBUG("%s: [DEBUG] " FMT "(%s:%d:%s)\n", TAG, ##__VA_ARGS__, __FILE__, __LINE__, __func__)
    
#define EOSL_LOG_DEBUG_S(TAG, FMT, ...)                                        \
    GGML_LOG_DEBUG("%s: [DEBUG] " FMT "\n", TAG, ##__VA_ARGS__);
