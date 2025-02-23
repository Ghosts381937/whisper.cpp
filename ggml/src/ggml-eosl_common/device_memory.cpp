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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-eosl_common/device_memory.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wimport-preprocessor-directive-pedantic"
#import "ggml-eosl_host/host_debug.hpp"
#pragma clang diagnostic pop
#else
#include "ggml-eosl_host/host_debug.hpp"
#endif

#define TAG "EOSL_MEMORY"

#define MINIMUM_BLOCK_SIZE   ((size_t)(128)) /* Block sizes must not get too small. */
#define HEAP_SIZE_MAX        (~((size_t)0))

eosl_dev_memory_heap::eosl_dev_memory_heap()    {
}

void eosl_dev_memory_heap::define_heap_regions(const eosl_heap_region * const heap_regions)    {
    block_link *current_block = nullptr;
    size_t region_size, total_heap_size = 0;
    uint8_t *aligned_start_address;
    int index = 0;
    const eosl_heap_region *heap_region;

    heap_region = &(heap_regions[index]);
    while(heap_region->size_in_bytes > 0)    {
        region_size = heap_region->size_in_bytes;

        /* Ensure the heap region starts on a correctly aligned boundary. */
        aligned_start_address = (uint8_t *)heap_region->start_address;
        if((((size_t)aligned_start_address) & BYTE_ALIGNMENT_MASK) != 0)    {
            aligned_start_address += (BYTE_ALIGNMENT - 1);
            aligned_start_address = (uint8_t *)(((size_t)aligned_start_address) & ~((size_t)BYTE_ALIGNMENT_MASK));

            /* Adjust the size for the bytes lost to alignment. */
            region_size -= (size_t)(aligned_start_address - heap_region->start_address);

            /* Adjust the size to alignment. */
            region_size &= ~((size_t)BYTE_ALIGNMENT_MASK);
        }

         /* Set m_root_block if it has not already been set. */
        if(index == 0)    {
            /* m_root_block is used to hold a pointer to the first item in the list of free blocks. */
            m_root_block.start_address= nullptr;
            m_root_block.block_size_in_bytes = (size_t)0;
            m_root_block.allocated = true;
            m_root_block.next_block = (block_link *)malloc(sizeof(block_link));
            current_block = m_root_block.next_block;
            current_block->next_block = nullptr;
            GGML_ASSERT(current_block != nullptr);
        } else    {
            /* Should only get here if one region has already been added to the heap. */
            GGML_ASSERT(current_block != nullptr);

            /* Check blocks are passed in with increasing start addresses. */
            GGML_ASSERT((size_t)aligned_start_address >
                        (size_t)(current_block->start_address + current_block->block_size_in_bytes));
            
            current_block->next_block = (block_link *)malloc(sizeof(block_link));
            GGML_ASSERT(current_block->next_block != nullptr);
            current_block = current_block->next_block;
            current_block->next_block = nullptr;
        }
        
        current_block->block_size_in_bytes = region_size;
        current_block->start_address = aligned_start_address;
        current_block->allocated = false;
        
        total_heap_size += current_block->block_size_in_bytes;
        
        /* Move onto the next HeapRegion_t structure. */
        index++;
        heap_region = &(heap_regions[index]);
    } // end of while
    
    m_minimum_ever_free_bytes_remaining = total_heap_size;
    m_free_bytes_remaining = total_heap_size;
    m_total_bytes = total_heap_size;
    m_initialized = true;

    /* Check something was actually defined before it is accessed. */
    GGML_ASSERT(total_heap_size);
}

void* eosl_dev_memory_heap::mallocc(size_t size)    {
    block_link *current_block;
    block_link *previous_block;
    block_link *new_block;
    size_t real_size = size, addition_size;
    void* result;
    
    /* The heap must be initialised before the first call to malloc */
    GGML_ASSERT(m_initialized);
    
    if(size <= 0)    GGML_ASSERT(false);
    
    /* Ensure that blocks are always aligned to the required number of bytes. */
    if((real_size & BYTE_ALIGNMENT_MASK) != 0x00)    {
        /* Byte alignment required. */
        addition_size = BYTE_ALIGNMENT - (real_size & BYTE_ALIGNMENT_MASK);
        real_size += addition_size;
    }
    
    if(real_size > m_free_bytes_remaining)    {
        EOSL_LOG_ERROR(TAG, "Device out of heap memory error.");
        GGML_ASSERT(false);
    }
        
    /* Traverse the list from the start (lowest address) block until
     * one of adequate size is found. */
    previous_block = &m_root_block;
    current_block = m_root_block.next_block;
    while((current_block != NULL) &&
          (current_block->block_size_in_bytes < real_size))    {
        previous_block = current_block;
        current_block = current_block->next_block;
    }
        
    if(current_block == NULL)    {
        EOSL_LOG_ERROR(TAG, "No free block found that fits %zu bytes", size);
        GGML_ASSERT(false);
    }

    if(current_block->block_size_in_bytes - real_size >= MINIMUM_BLOCK_SIZE)    {
        new_block = (block_link *)malloc(sizeof(block_link));
        GGML_ASSERT(new_block != nullptr);
        result = (void *)(current_block->start_address + current_block->block_size_in_bytes - real_size);
        new_block->start_address = current_block->start_address;
        new_block->block_size_in_bytes = current_block->block_size_in_bytes - real_size;
        new_block->allocated = false;
        current_block->block_size_in_bytes = real_size;
        current_block->start_address = (uint8_t *)result;
        current_block->allocated = true;
        previous_block->next_block = new_block;
        new_block->next_block = current_block;
        //EOSL_LOG_DEBUG_S(TAG, "%s: 0x%zx/0x%zx/0x%zx", __func__, size, real_size, new_block->block_size_in_bytes);
    } else    {
        result = (void *)current_block->start_address;
        real_size = current_block->block_size_in_bytes;
        current_block->allocated = true;
    }
        
    m_free_bytes_remaining -= real_size;
    if(m_free_bytes_remaining < m_minimum_ever_free_bytes_remaining)    {
        m_minimum_ever_free_bytes_remaining = m_free_bytes_remaining;
    }
    m_number_of_successful_allocations++;

    return result;
}

static void collapse_free_blocks(block_link *block)    {
    block_link *previous_block, *current_block;
    
    previous_block = block;
    while((current_block = previous_block->next_block) != NULL)    {
        if((previous_block->allocated == false) &&
           (current_block->allocated == false))    {
            previous_block->block_size_in_bytes += current_block->block_size_in_bytes;
            previous_block->next_block = current_block->next_block;
            free(current_block);
        } else    {
            break;
        }
    }
}
        
void eosl_dev_memory_heap::freee(void *data)    {
    block_link *previous_block, *current_block;

    if(data == NULL)    {
        EOSL_LOG_ERROR(TAG, "Invalid, freeing nullptr");
        return;
    }
    
    GGML_ASSERT(m_initialized);
    
    previous_block = &m_root_block;
    current_block = previous_block->next_block;
    while(current_block != NULL)    {/* Max value that fits in a size_t type. */
        if((current_block->start_address == (uint8_t *)data) && (current_block->allocated == true))    {
            current_block->allocated = false;
            m_number_of_successful_frees++;
            m_free_bytes_remaining += current_block->block_size_in_bytes;
            
            if(previous_block->allocated)
                collapse_free_blocks(current_block);
            else
                collapse_free_blocks(previous_block);
            
            return;
        }
         
        previous_block = current_block;
        current_block = current_block->next_block;
    }
    
    EOSL_LOG_ERROR(TAG, "Unable to free 0x%p, invalid address", data);
}

size_t eosl_dev_memory_heap::get_size_of_largest_free_block_in_bytes()    {
    block_link *current_block;
    size_t size = 0;
    
    current_block = m_root_block.next_block;
    while(current_block != NULL)    {
        if(current_block->allocated == false)
            if(size < current_block->block_size_in_bytes)
                size = current_block->block_size_in_bytes;
                
        current_block = current_block->next_block;
    }
    
    return size;
}

size_t eosl_dev_memory_heap::get_size_of_smallest_free_block_in_bytes()    {
    block_link *current_block;
    size_t size = HEAP_SIZE_MAX;
    
    current_block = m_root_block.next_block;
    while(current_block != NULL)    {
        if(current_block->allocated == false)
            if(size > current_block->block_size_in_bytes)
                size = current_block->block_size_in_bytes;
                
        current_block = current_block->next_block;
    }
    
    return size;
}

size_t eosl_dev_memory_heap::get_number_of_free_blocks()    {
    block_link *current_block;
    size_t count = 0;
    
    current_block = m_root_block.next_block;
    while(current_block != NULL)    {
        if(current_block->allocated == false)    count++;
        current_block = current_block->next_block;
    }
    
    return count;
}

void eosl_dev_memory_heap::reset_heap_minimum_ever_free_heap_size()    {
    m_minimum_ever_free_bytes_remaining = m_free_bytes_remaining;
}

/*
 * Reset the state in this file. This state is normally initialized at start up.
 * This function must be called by the application before restarting the
 * scheduler.
 */
void eosl_dev_memory_heap::heap_reset_state(void)    {
    block_link *tmp, *current_block;
    
    m_initialized = false;

    m_free_bytes_remaining = (size_t)0U;
    m_minimum_ever_free_bytes_remaining = (size_t)0U;
    m_number_of_successful_allocations = (size_t)0U;
    m_number_of_successful_frees = (size_t)0U;
    
    current_block = m_root_block.next_block;
    while(current_block != NULL)    {
        tmp = current_block;
        free(current_block);
        current_block = tmp->next_block;
    }
}

eosl_dev_memory_heap::~eosl_dev_memory_heap()    {
    block_link *tmp, *current_block;
    
    current_block = m_root_block.next_block;
    while(current_block != NULL)    {
        tmp = current_block;
        free(current_block);
        current_block = tmp->next_block;
    }
}

void eosl_dev_memory_heap::report()    {
    char buffer[2048];
    int count = 0;
    
    count += snprintf(buffer + count, sizeof(buffer) - count,
                      "available_heapspace_in_bytes: 0x%zx\n", get_available_heapspace_in_bytes());
    count += snprintf(buffer + count, sizeof(buffer) - count,
                      "size_of_largest_free_block_in_bytes: 0x%zx\n", get_size_of_largest_free_block_in_bytes());
    count += snprintf(buffer + count, sizeof(buffer) - count,
                      "number_of_free_blocks: %zu\n", get_number_of_free_blocks());
    EOSL_LOG_INFO(TAG, "\n%s\n", buffer);
}



        
        
          
            


    


        
        
        
        


