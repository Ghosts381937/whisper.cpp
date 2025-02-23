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

#include <time.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wimport-preprocessor-directive-pedantic"
#import "ggml-eosl_host/host_debug.hpp"
#pragma clang diagnostic pop
#else
#include "ggml-eosli_host/host_debug.hpp"
#endif

#include "ggml-eosl_host/host_utils.hpp"
#include "ggml-eosl_host/host_system.hpp"
#include "xdma/cdev_sgdma.h"

#define TAG "DEVICE_DMA"

#define VERBOSE      0
#define RW_MAX_SIZE  0x7ffff000

int ggml_eosl_host_backend_dev::trigger_irq_to_dev()    {
    uint32_t irq = 1 << 3;
    
    return h2c_devmem2(nullptr, dev_info.pcie_reg_space_host, 'w', irq);
}

int ggml_eosl_host_backend_dev::wait_for_irq_from_dev()    {
    int ret;
    int irqs;
    struct pollfd pfd;
    
    if(event_c2h == 0)    {
        if((event_c2h = open(dev_info.event_c2h_dev_path, O_RDONLY | O_CLOEXEC)) == -1)    {
            EOSL_LOG_ERROR(TAG, "Error opening %s: %d, %s", dev_info.event_c2h_dev_path, errno, strerror(errno));
            GGML_ASSERT(false);
        }
    }
    
    while(true)    {   
        pfd.fd = event_c2h;
        pfd.events = POLLIN;
        pfd.revents = 0;

        ret = poll(&pfd, 1, 256);
        if(ret == -1)    {   
            EOSL_LOG_ERROR(TAG, "Error polling %s: %s", dev_info.event_c2h_dev_path, strerror(errno));
            GGML_ASSERT(false);
        } else if(ret)    { // Data is available
            if(4 != read(event_c2h, (void *)&irqs, 4))    {   
                EOSL_LOG_ERROR(TAG, "Error reading %s: %s", dev_info.event_c2h_dev_path, strerror(errno));
                GGML_ASSERT(false);
            }
    
            break;
        } else    {
            continue;
        }
    }   

    return irqs;
}

int ggml_eosl_host_backend_dev::clear_irq_from_dev()    {
    uint32_t irq = 1 << 3;
    
    return h2c_devmem2(nullptr, (void *)((uint8_t *)(dev_info.pcie_reg_space_host) + 4), 'w', irq);
}

static int timespec_check(struct timespec *t);
static void timespec_sub(struct timespec *t1, struct timespec *t2);
static ssize_t write_from_buffer(const char *fname, const int fd,
                                 void *src, size_t size, size_t base);
static ssize_t read_to_buffer(const char *fname, const int fd,
                              void *buffer, uint64_t size, size_t base);

ssize_t ggml_eosl_host_backend_dev::
dma_from_device(void *device_ptr, void *data, size_t aperture, size_t size, size_t offset)    {
    ssize_t rc = 0;
    size_t bytes_done = 0;
    uint8_t  *buffer = NULL;
    struct timespec ts_start, ts_end;
    long total_time = 0;
    float result, avg_time = 0;
    int underflow = 0;

    if(c2h_fd == 0)    {
        c2h_fd = open(dev_info.dma_c2h_dev_path, O_RDWR | O_CLOEXEC);
        if(c2h_fd < 0) {
            EOSL_LOG_ERROR(TAG, "Unable to open device %s, %s.", dev_info.dma_c2h_dev_path, strerror(errno));
            GGML_ASSERT(false);
        }
    }

    buffer = ((uint8_t *)data) + offset;
    rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
    if(aperture) {
        struct xdma_aperture_ioctl io;
        io.buffer = (unsigned long)buffer;
        io.len = size;
        io.ep_addr = (uint64_t)device_ptr;
        io.aperture = aperture;
        io.done = 0UL;
        rc = ioctl(c2h_fd, IOCTL_XDMA_APERTURE_R, &io);
        if(rc < 0 || io.error) {
            EOSL_LOG_ERROR(TAG, "Aperture R failed %zd,%d.", rc, io.error);
            goto out;
        }

        bytes_done = io.done;
    } else {
        rc = read_to_buffer(dev_info.dma_c2h_dev_path, c2h_fd,
                            (void *)buffer, size, (size_t)device_ptr);
        if(rc < 0)
            goto out;
        
        bytes_done = rc;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    if(bytes_done < size) {
        EOSL_LOG_ERROR(TAG, "Underflow %ld/%ld.", bytes_done, size);
        underflow = 1;
    }

    /* subtract the start time from the end time */
    timespec_sub(&ts_end, &ts_start);
    total_time += ts_end.tv_nsec;
                
    /* a bit less accurate but side-effects are accounted for */
    if(VERBOSE)
        EOSL_LOG_INFO(TAG, "CLOCK_MONOTONIC %ld.%09ld sec. read %ld/%ld bytes",
                      ts_end.tv_sec, ts_end.tv_nsec, bytes_done, size);

    if(!underflow) {
        avg_time = (float)total_time;
        result = ((float)size) * 1000 / avg_time;
        if(VERBOSE)
            EOSL_LOG_INFO(TAG, "** Avg time device %s, total time %ld nsec, avg_time = %f, size = %lu, BW = %f",
                          dev_info.dma_c2h_dev_path, total_time, avg_time, size, result);
        //EOSL_LOG_INFO(TAG, "%s ** Average BW = %lu, %f", dev_info.dma_c2h_dev_path, size, result);
        rc = (ssize_t)bytes_done;
    } else    {
        rc = -EIO;
    }
    
out:

    return rc;
}

ssize_t ggml_eosl_host_backend_dev::
dma_to_device(void *device_ptr, const void *data, size_t aperture, size_t size, size_t offset)    {
    ssize_t rc;
    size_t bytes_done = 0;
    uint8_t *buffer = ((uint8_t *)data) + offset;
    struct timespec ts_start, ts_end;
    long total_time = 0;
    float result, avg_time = 0;
    int underflow = 0;

    if(h2c_fd == 0) {
        h2c_fd = open(dev_info.dma_h2c_dev_path, O_RDWR | O_CLOEXEC);
        if(h2c_fd < 0)    {
            EOSL_LOG_ERROR(TAG, "Unable to open device %s, %s.",
                           dev_info.dma_h2c_dev_path, strerror(errno));
            GGML_ASSERT(false);
        }
    }

    /* write buffer to AXI MM address using SGDMA */
    rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
    if(aperture) {
        struct xdma_aperture_ioctl io;
        io.buffer = (unsigned long)buffer;
        io.len = size;
        io.ep_addr = (uint64_t)device_ptr;
        io.aperture = aperture;
        io.done = 0UL;

        rc = ioctl(h2c_fd, IOCTL_XDMA_APERTURE_W, &io);
        if(rc < 0 || io.error) {
            EOSL_LOG_ERROR(TAG, "aperture W ioctl failed %zd, %d.", rc, io.error);
            goto out;
        }

        bytes_done = io.done;
    } else {
        rc = write_from_buffer(dev_info.dma_h2c_dev_path, h2c_fd,
                               (void *)buffer, size, (size_t)device_ptr);
        
        if(rc < 0) goto out;
        
        bytes_done = rc;
    }
    rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);

    if(bytes_done < size) {
        EOSL_LOG_ERROR(TAG, "underflow %ld/%ld.",bytes_done, size);
        underflow = 1;
    }

    /* subtract the start time from the end time */
    timespec_sub(&ts_end, &ts_start);
    total_time += ts_end.tv_nsec;
    /* a bit less accurate but side-effects are accounted for */
    if(VERBOSE)
        EOSL_LOG_INFO(TAG, "CLOCK_MONOTONIC %ld.%09ld sec. write %ld bytes",
                      ts_end.tv_sec, ts_end.tv_nsec, size);

    if(!underflow)    {
        avg_time = (float)total_time;
        result = ((float)size) * 1000 / avg_time;
        if(VERBOSE)
            EOSL_LOG_INFO(TAG, "** Avg time device %s, total time %ld nsec, avg_time = %f, size = %lu, BW = %f",
                          dev_info.dma_h2c_dev_path, total_time, avg_time, size, result);
        //EOSL_LOG_INFO(TAG, "%s ** Average BW = %lu, %f", dev_info.dma_h2c_dev_path, size, result);
        rc = (ssize_t)bytes_done;
    } else    {
        rc = -EIO;
    }

out:

    return rc;
}

static ssize_t read_to_buffer(const char *fname, const int fd,
                              void *buffer, uint64_t size, size_t base)    {
    ssize_t rc;
    size_t count = 0;
    uint8_t *buf = (uint8_t *)buffer;
    off_t offset = base;
    int loop = 0;

    while(count < size) {
        uint64_t bytes = size - count;

        if(bytes > RW_MAX_SIZE)
            bytes = RW_MAX_SIZE;

        if(offset) {
            rc = lseek(fd, offset, SEEK_SET);
            if(rc != offset) {
                EOSL_LOG_ERROR(TAG, "%s, seek off 0x%lx != 0x%lx.", fname, rc, offset);
                return -EIO;
            }
        }

        /* read data from file into memory buffer */
        rc = read(fd, buf, bytes);
        if(rc < 0) {
            EOSL_LOG_ERROR(TAG, "%s, read 0x%lx @ 0x%lx failed, %s.", fname, bytes, offset, strerror(errno));
            return -EIO;
        }

        count += rc;
        if((size_t)rc != bytes) {
            EOSL_LOG_ERROR(TAG, "%s, read underflow 0x%lx/0x%lx @ 0x%lx.", fname, rc, bytes, offset);
            break;
        }
                
        buf += bytes;
        offset += bytes;
        loop++;
    }

    if(count != size && loop)
        EOSL_LOG_ERROR(TAG, "%s, read underflow 0x%lx/0x%lx.", fname, count, size);
    
    return count;
}

static ssize_t write_from_buffer(const char *fname, const int fd,
                                void *src, size_t size, size_t base)    {
    ssize_t rc; 
    size_t count = 0;
    uint8_t *buf = (uint8_t *)src;
    off_t offset = base;
    int loop = 0;

    while(count < size) {
        size_t bytes = size - count;
        if(bytes > RW_MAX_SIZE)
            bytes = RW_MAX_SIZE;

        if(offset) {
            rc = lseek(fd, offset, SEEK_SET);
            if(rc != offset) {
                EOSL_LOG_ERROR(TAG, "%s, seek off 0x%lx != 0x%lx.", fname, rc, offset);
                return -EIO;
            }
        }

        /* write data to file from memory buffer */
        rc = write(fd, buf, bytes);
        if(rc < 0) {
            EOSL_LOG_ERROR(TAG, "%s, write 0x%lx @ 0x%lx failed %ld.", fname, bytes, offset, rc);
            return -EIO;
        }

        count += (size_t)rc; 
        if((size_t)rc != bytes) {
            EOSL_LOG_ERROR(TAG,"%s, write underflow 0x%lx/0x%lx @ 0x%lx.\n", fname, rc, bytes, offset);
            break;
        }
        buf += bytes;
        offset += bytes;
        
        loop++;
    }
        
    if(count != size && loop)
        EOSL_LOG_ERROR(TAG, "%s, write underflow 0x%lx/0x%lx.\n", fname, count, size);
        
    fsync(fd);

    return count;
}

/* Subtract timespec t2 from t1
 *
 * Both t1 and t2 must already be normalized
 * i.e. 0 <= nsec < 1000000000
 */
static int timespec_check(struct timespec *t)    {
    if((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
        return -1; 
        
    return 0;
}

static void timespec_sub(struct timespec *t1, struct timespec *t2)    {
    if(timespec_check(t1) < 0) {
        EOSL_LOG_ERROR(TAG, "invalid time #1: %lld.%.9ld.\n", (long long)t1->tv_sec, t1->tv_nsec);
        return;
    }
    
    if(timespec_check(t2) < 0) {
        EOSL_LOG_ERROR(TAG, "invalid time #2: %lld.%.9ld.\n", (long long)t2->tv_sec, t2->tv_nsec);
        return;
    }
    
    t1->tv_sec -= t2->tv_sec;
    t1->tv_nsec -= t2->tv_nsec;
    if (t1->tv_nsec >= 1000000000) {
        t1->tv_sec++;
        t1->tv_nsec -= 1000000000;
    } else if (t1->tv_nsec < 0) {
        t1->tv_sec--;
        t1->tv_nsec += 1000000000;
    }
}
