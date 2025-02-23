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

#include <array>
#include <string>

#include "ggml.h"
#include "ggml-eosl_host.h"
#include "ggml-eosl_common.h"
#include "ggml-eosl_common/device_memory.hpp"

void* ggml_eosl_host_malloc(size_t size);
void ggml_eosl_host_free(void* ptr);

struct ggml_eosl_dev_info    {
    size_t buffer_start;
    size_t buffer_size;

    void   *pcie_reg_space_host;
    char   dma_h2c_dev_path[GGML_MAX_NAME << 1];
    char   dma_c2h_dev_path[GGML_MAX_NAME << 1];
    char   event_c2h_dev_path[GGML_MAX_NAME << 1];
};

class ggml_eosl_host;

class ggml_eosl_host_backend_dev    {
private:
    int dev_id;
    std::string name;
    ggml_eosl_dev_info dev_info;
    eosl_dev_memory_heap dev_memory_heap;
    
    int h2c_fd = 0;
    int c2h_fd = 0;
    int event_c2h = 0;
    
    ggml_eosl_host_backend_dev(int id, size_t buffer_start, size_t buffer_size,
                               void *pcie_reg_space, const char *h2c_dev,
                               const char *c2h_dev, const char *event_c2h_dev);

public:
    ~ggml_eosl_host_backend_dev() = default;
    
    int get_id() { return dev_id; }
    const char *get_name() { return name.c_str(); }
    void get_device_memory(size_t *free, size_t *total);
    void *malloc_memory(size_t size);
    void free_memory(void *ptr);
    
    // TODO:  aperture==1 does NOT work...
    ssize_t dma_to_device(void *device_ptr, const void *data, size_t aperture, size_t size, size_t offset);
    ssize_t dma_from_device(void *device_ptr, void *data, size_t aperture, size_t size, size_t offset);
    int trigger_irq_to_dev();
    int wait_for_irq_from_dev();
    int clear_irq_from_dev();
    
    int send_eosl_dev_cmd(enum ggml_eosl_dev_cmd cmd, ...);
    
    friend class ggml_eosl_host;
};
    
class ggml_eosl_host    {
private:
    int device_count;

    ggml_eosl_host_backend_dev *devices[GGML_EOSL_MAX_DEVICES]; 
    std::array<float, GGML_EOSL_MAX_DEVICES> default_tensor_split = {}; 
    int max_work_group_sizes[GGML_EOSL_MAX_DEVICES] = {0};
    
    static ggml_guid s_guid;
    static ggml_eosl_host *s_ggml_eosl_host;
private:
    ggml_eosl_host();
    
public:
    ~ggml_eosl_host();
    
    int get_device_count() { return device_count; }
    ggml_guid_t get_backend_guid() { return &s_guid; }
    void check_allow_gpu_index(const int device_index);
    ggml_eosl_host_backend_dev *get_backend_device(int device);
    
    static ggml_eosl_host *get_ggml_eosl_host(); // Singleton
};

ggml_guid_t ggml_eosl_get_backend_guid();

