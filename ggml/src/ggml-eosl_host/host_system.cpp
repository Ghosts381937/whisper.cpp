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

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

#include "device_config.hpp"
#include "host_system.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wimport-preprocessor-directive-pedantic"
#import "ggml-eosl_host/host_debug.hpp"
#pragma clang diagnostic pop
#else
#include "ggml-eosl_host/host_debug.hpp"
#endif

#define TAG "EOSL_HOST_SYSTEM"

ggml_eosl_host_backend_dev::
ggml_eosl_host_backend_dev(int id, size_t buffer_start, size_t buffer_size,
                           void *pcie_reg_space, const char *h2c_dev,
                           const char *c2h_dev, const char *event_c2h_dev)    {
    dev_id = id;
    name = GGML_EOSL_NAME "-BACKEND-DEVICE-" + std::to_string(id);
    dev_info.buffer_start = buffer_start;
    dev_info.buffer_size = buffer_size;
    dev_info.pcie_reg_space_host = pcie_reg_space;
    strncpy(dev_info.dma_h2c_dev_path, h2c_dev, GGML_MAX_NAME << 1);
    strncpy(dev_info.dma_c2h_dev_path, c2h_dev, GGML_MAX_NAME << 1);
    strncpy(dev_info.event_c2h_dev_path, event_c2h_dev, GGML_MAX_NAME << 1);
 
    const eosl_heap_region regions[] =    {
        { (uint8_t *)(buffer_start), buffer_size },
        { (uint8_t *)NULL, 0 } /* Terminates the array. */
    };

    dev_memory_heap.define_heap_regions(regions);
}

void ggml_eosl_host_backend_dev::get_device_memory(size_t *free, size_t *total)    {
    *total = dev_memory_heap.get_total_heapspace_in_bytes();
    *free = dev_memory_heap.get_available_heapspace_in_bytes();
}

void* ggml_eosl_host_backend_dev::malloc_memory(size_t size)    {
    return dev_memory_heap.mallocc(size);
}

void ggml_eosl_host_backend_dev::free_memory(void *ptr)    {
    dev_memory_heap.freee(ptr);
}

ggml_eosl_host_backend_dev *get_backend_device(int device);

ggml_guid ggml_eosl_host::s_guid = {
    0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d,
    0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x54 };

ggml_eosl_host* ggml_eosl_host::s_ggml_eosl_host = nullptr;

ggml_eosl_host::ggml_eosl_host()    {
    int i;

    //TODO: Scan EOSL devices
    device_count = 1;
    GGML_ASSERT(device_count <= GGML_EOSL_MAX_DEVICES);
    for(i = 0; i < device_count; i++)    {
        devices[i] = new ggml_eosl_host_backend_dev(i,
                                                    DEVICE_BUFFER_START, DEVICE_BUFFER_SIZE,
                                                    (void *)(0X86000000), /* PCIe */
                                                    "/dev/xdma0_h2c_0",
                                                    "/dev/xdma0_c2h_0",
                                                    "/dev/xdma0_events_3");
    }
}

ggml_eosl_host_backend_dev* ggml_eosl_host::get_backend_device(int device)    {
    return devices[device];
}

ggml_eosl_host* ggml_eosl_host::get_ggml_eosl_host()    {
    if(nullptr == s_ggml_eosl_host)
        s_ggml_eosl_host = new ggml_eosl_host();

    return s_ggml_eosl_host;
}

/*
 * device_index: device index from 0 to n (continue numbers).
 *   It is used for device select/set in EOSL backend internal data structure.
 */
void ggml_eosl_host::check_allow_gpu_index(const int device_index) {
    if(device_index >= device_count) {
        char error_buf[256];
        snprintf(error_buf, sizeof(error_buf),
                 "%s error: device_index:%d is out of range: [0-%d]",
                 __func__, device_index, device_count - 1);
        EOSL_LOG_ERROR_S(TAG, "%s", error_buf);
        assert(false);
    }
}

static inline int get_eosl_host_env(const char *env_name, int default_val) {
    char *user_device_string = getenv(env_name);
    int user_number = default_val;

    unsigned n;
    if (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int)n;
    } else {
        user_number = default_val;
    }    
    return user_number;
}

ggml_guid_t ggml_eosl_get_backend_guid()    {
    return ggml_eosl_host::get_ggml_eosl_host()->get_backend_guid();
}
