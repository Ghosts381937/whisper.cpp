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
#include <stdarg.h>
#include <unistd.h>
#include <uuid/uuid.h>

#include "ggml.h"
#include "host_system.hpp"
#include "device_config.hpp"
#include "host_utils.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wimport-preprocessor-directive-pedantic"
#import "ggml-eosl_host/host_debug.hpp"
#pragma clang diagnostic pop
#else
#include "ggml-eosl_host/host_debug.hpp"
#endif

#define TAG "EOSL_HOST_MSG"

int ggml_eosl_host_backend_dev::
send_eosl_dev_cmd(enum ggml_eosl_dev_cmd cmd, ...)    {
    va_list arguments;                     
    uint8_t buffer[512];
    uuid_t binuuid;
    char uuid[37];
    size_t index;
    int result;
    void *output;
    size_t output_size;
    void *device_ptr;
    uint64_t result_addr, result_size;
    
    // get backend device
    ggml_eosl_host_backend_dev *device = ggml_eosl_host::get_ggml_eosl_host()->get_backend_device(dev_id);
    
    // typedef uint8_t uuid_t[16];
    uuid_generate_random(binuuid);
    uuid_unparse_upper(binuuid, uuid);
    for(index = 0; index < sizeof(uuid_t); index++)
        buffer[index] = binuuid[index];
        
    *((uint32_t *)(buffer + index)) = (uint32_t)cmd;
    index += sizeof(uint64_t);
    
    va_start(arguments, cmd);
    
    switch(cmd)    {
        case EOSL_GGML_DEV_CMD_ALLOC_BUFFER:    {
            uint64_t backend_buffer = va_arg(arguments, uint64_t); 
            *((uint64_t *)(buffer + index)) = backend_buffer;
            index += sizeof(uint64_t);
            uint64_t device_ptr = va_arg(arguments, uint64_t);
            *((uint64_t *)(buffer + index)) = device_ptr;
            index += sizeof(uint64_t);
            uint64_t size = va_arg(arguments, uint64_t);
            *((uint64_t *)(buffer + index)) = size;
            index += sizeof(uint64_t);
            EOSL_LOG_DEBUG_S(TAG, "Sending command to device: %s, %d,  0x%" PRIx64 ", 0x%" PRIx64 ", 0x%" PRIx64,
                             uuid, cmd, backend_buffer, device_ptr, size);
            break;
        }
        case EOSL_GGML_DEV_CMD_FREE_BUFFER:    {   
            uint64_t backend_buffer = va_arg(arguments, uint64_t); 
            *((uint64_t *)(buffer + index)) = backend_buffer;
            index += sizeof(uint64_t);
            uint64_t device_ptr = va_arg(arguments, uint64_t);
            *((uint64_t *)(buffer + index)) = device_ptr;
            index += sizeof(uint64_t);
            uint64_t size = va_arg(arguments, uint64_t);
            *((uint64_t *)(buffer + index)) = size;
            index += sizeof(uint64_t);

            EOSL_LOG_DEBUG_S(TAG, "Sending command to device: %s, %d,  0x%" PRIx64 ", 0x%" PRIx64 ", 0x%" PRIx64,
                             uuid, cmd, backend_buffer, device_ptr, size);

            break;
        }
        case EOSL_GGML_DEV_CMD_INIT_TENSOR:    {
            void *input = (void *)va_arg(arguments, uint64_t);
            size_t input_size = (size_t)va_arg(arguments, uint64_t);
            
            device_ptr = device->malloc_memory(input_size);
            dma_to_device(device_ptr, input, 0, input_size, 0);
          
            *((uint64_t *)(buffer + index)) = (uint64_t)device_ptr;
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = (uint64_t)input_size;
            index += sizeof(uint64_t);
            
            EOSL_LOG_DEBUG_S(TAG, "Sending command to device: %s, %d, %p, 0x%" PRIx64,
                             uuid, cmd, device_ptr, input_size);
            
            break;
        }
        case EOSL_GGML_DEV_CMD_SET_TENSOR:    {
            uint64_t buffer_id = va_arg(arguments, uint64_t);
            uint64_t id = va_arg(arguments, uint64_t);
            device_ptr = (void *)va_arg(arguments, uint64_t);
            void *input = (void *)va_arg(arguments, uint64_t);
            size_t offset = (size_t)va_arg(arguments, uint64_t);
            size_t input_size = (size_t)va_arg(arguments, uint64_t);
            
            device->dma_to_device(((uint8_t *)device_ptr) + offset, input, 0, input_size, 0);
            
            *((uint64_t *)(buffer + index)) = buffer_id;
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = id;
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = (uint64_t)device_ptr;
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = (uint64_t)input_size;
            index += sizeof(uint64_t);
            
            EOSL_LOG_DEBUG_S(TAG, "Sending command to device: %s, %d, 0x%" PRIx64  ", 0x%" PRIx64 ", %p, 0x%" PRIx64,
                             uuid, cmd, buffer_id, id, device_ptr, input_size);
            
            break;
        }
        case EOSL_GGML_DEV_CMD_BUFFER_CLEAR:
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // address
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // size
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // value
            index += sizeof(uint64_t);
            break;
        case EOSL_GGML_DEV_CMD_MEMSET_TENSOR:
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // buffer id (address) 
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // tensor id (address)
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // value
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // offset
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // size
            index += sizeof(uint64_t);
            break;
        case EOSL_GGML_DEV_CMD_COPY_TENSOR:
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // buffer id (address)
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // tensor(src) id (address)
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = va_arg(arguments, uint64_t); // tensor(dst) id (address) 
            index += sizeof(uint64_t);
            break;
        case EOSL_GGML_DEV_CMD_GRAPH_COMPUTE:    {
            void *input = (void *)va_arg(arguments, uint64_t);
            size_t input_size = (size_t)va_arg(arguments, uint64_t);
            
            device_ptr = device->malloc_memory(input_size);
            dma_to_device(device_ptr, input, 0, input_size, 0);
          
            *((uint64_t *)(buffer + index)) = (uint64_t)device_ptr;
            index += sizeof(uint64_t);
            *((uint64_t *)(buffer + index)) = (uint64_t)input_size;
            index += sizeof(uint64_t);
            
            EOSL_LOG_DEBUG_S(TAG, "Sending command to device: %s, %d, %p, 0x%" PRIx64,
                             uuid, cmd, device_ptr, input_size);
            
            break;
        }
        default:
            break;
    }
    
    output = (void *)va_arg(arguments, uint64_t);
    output_size = (size_t)va_arg(arguments, uint64_t);
    
    va_end(arguments); // Cleans up the list
    // Sending arguments
    dma_to_device((void *)(HOST_ARGS_START + 1024 * 3), (void *)buffer, 0, sizeof(buffer), 0);
    
    trigger_irq_to_dev(); // Trigering IRQ to MCU
    wait_for_irq_from_dev();
    clear_irq_from_dev(); // Clear IRQ from MCU

    switch(cmd)    {
        case EOSL_GGML_DEV_CMD_INIT_TENSOR:
        case EOSL_GGML_DEV_CMD_GRAPH_COMPUTE:
            // Release data
            device->free_memory((void *)device_ptr);
            break;
        default:
            break;
    }
    
    dma_from_device((void *)(HOST_ARGS_START + 1024 * 3 + 512), (void *)buffer, 0, sizeof(buffer), 0);
    for(index = 0; index < sizeof(uuid_t); index++)
        binuuid[index] = buffer[index];
    uuid_unparse_upper(binuuid, uuid);
    result = *((int *)(buffer + index));
    index += sizeof(uint64_t);
    
    if(output == nullptr)    return result;
    
    result_addr = *((uint64_t *)(buffer + index));
    index += sizeof(uint64_t);
    result_size = *((uint64_t *)(buffer + index));
    index += sizeof(uint64_t);
    EOSL_LOG_DEBUG_S(TAG, "Receiving result from device: %s, %d, 0X%" PRIX64 ", %" PRIu64,
                     uuid, result, result_addr, result_size);
                     
    if(output_size != result_size)
        EOSL_LOG_WARN_S(TAG, "Receiving result size id different from original output size: %" PRIu64 "/%zu",
                        result_size, output_size);
    
    if(result_size > 0)
        dma_from_device(output, (void *)result_addr, 0, result_size, 0);
    
    return result;
}
