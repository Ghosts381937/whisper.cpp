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
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

#include "ggml.h"
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

#define TAG "EOSL_HOST_UTILS"

#define MAP_SIZE 4096ULL
#define MAP_MASK (MAP_SIZE - 1)

static int devmem2(bool read, void *result, void *addr, char access_type, uint64_t write_value)    {
    static int fd = -1;
    static off_t target = 0;
    static void *map_base = nullptr;
    void *virt_addr;
    
    if(fd < 0)    {
        if((fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC)) == -1)    {
            EOSL_LOG_ERROR(TAG, "Error opening /dev/mem: %d, %s", errno, strerror(errno));
            GGML_ASSERT(false);
        }
    }

    if(target != (off_t)addr)    {
        if(nullptr != map_base)    {
            if(munmap(map_base, MAP_SIZE) != 0)
                EOSL_LOG_ERROR(TAG, "Error unmmapping /dev/mem: %s", strerror(errno));

            map_base = nullptr;
        }

        target = (off_t)addr;
    }
        
    /* Map one page */
    if(nullptr == map_base)    {
        map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, target & ~MAP_MASK);
        if(map_base == (void *)-1)    { // MAP_FAILED
            EOSL_LOG_ERROR(TAG, "Error mmapping /dev/mem: %s", strerror(errno));
            GGML_ASSERT(false);
        }
    }
    
    virt_addr = (void *)(((uint8_t *)map_base) + (target & MAP_MASK));
    if(read & (result != nullptr))    {   
        switch(access_type)    {
            case 'b':
                WRITE(result, uint8_t, READ(virt_addr, uint8_t));
                return 1;
            case 'h':
                WRITE(result, uint16_t, READ(virt_addr, uint16_t));
                return 2;
            case 'w':
                WRITE(result, uint32_t, READ(virt_addr, uint32_t));
                return 4;
            case 'l':
                WRITE(result, uint64_t, READ(virt_addr, uint64_t));
                return 8;
            default:
                EOSL_LOG_ERROR_S(TAG, "Illegal read data type '%c'.\n", access_type);
                return 0;
        }
    } else if(!read)    {
        switch(access_type)    {
            case 'b':
                WRITE(virt_addr, uint8_t, write_value);
                return 1;
            case 'h':
                WRITE(virt_addr, uint16_t, write_value);
                return 2;
            case 'w':
                WRITE(virt_addr, uint32_t, write_value);
                return 4;
            case 'l':
                WRITE(virt_addr, uint64_t, write_value);
                return 8;
            default:
                EOSL_LOG_ERROR_S(TAG, "Illegal write data type '%c'.\n", access_type);
                return 0;
        }
    }
    
    return 0;
}

int c2h_devmem2(void *result, void *addr, char access_type)    {
    return devmem2(true, result, addr, access_type, 0);
}

int h2c_devmem2(void *result, void *addr, char access_type, uint64_t write_value)    {
    return devmem2(false, result, addr, access_type, write_value);
}
