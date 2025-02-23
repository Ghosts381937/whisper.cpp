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

#ifndef GGML_EOSL_MEMORY_HPP
#define GGML_EOSL_MEMORY_HPP

#include <inttypes.h>

#include "ggml-impl.h"

#define BYTE_ALIGNMENT             (TENSOR_ALIGNMENT)
#if BYTE_ALIGNMENT == 32
    #define BYTE_ALIGNMENT_MASK    ( 0x001f )
#elif BYTE_ALIGNMENT == 16
    #define BYTE_ALIGNMENT_MASK    ( 0x000f )
#elif BYTE_ALIGNMENT == 8
    #define BYTE_ALIGNMENT_MASK    ( 0x0007 )
#elif BYTE_ALIGNMENT == 4
    #define BYTE_ALIGNMENT_MASK    ( 0x0003 )
#elif BYTE_ALIGNMENT == 2
    #define BYTE_ALIGNMENT_MASK    ( 0x0001 )
#elif BYTE_ALIGNMENT == 1
    #define BYTE_ALIGNMENT_MASK    ( 0x0000 )
#else /* if BYTE_ALIGNMENT == 32 */
    #error "Invalid BYTE_ALIGNMENT definition"
#endif /* if portBYTE_ALIGNMENT == 32 */

struct eosl_heap_region    {
    uint8_t *start_address;
    size_t size_in_bytes;
};
typedef eosl_heap_region *eosl_heap_region_t;

/*
 * Define the linked list structure.  This is used to link free blocks in order of their memory address.
 */
struct block_link    {
    uint8_t *start_address;
    size_t block_size_in_bytes;               /**< The size of the free block. */
    bool allocated;
    struct block_link *next_block = nullptr; /**< The next block in the list. */
};
typedef block_link block_link_t;

class eosl_dev_memory_heap {
private:
    bool m_initialized = false;
    block_link m_root_block;
    
    /* Keeps track of the number of calls to allocate and free memory as well as the
     * number of free bytes remaining, but says nothing about fragmentation. */
    size_t m_total_bytes = (size_t)0U;
    size_t m_free_bytes_remaining = (size_t)0U;
    size_t m_minimum_ever_free_bytes_remaining = (size_t)0U;
    size_t m_number_of_successful_allocations = (size_t)0U;
    size_t m_number_of_successful_frees = (size_t)0U;

public:
    eosl_dev_memory_heap();
    ~eosl_dev_memory_heap();

    void define_heap_regions(const eosl_heap_region * const heap_regions);

    size_t get_total_heapspace_in_bytes() { return m_total_bytes; }
    /* The total heap size currently available - this is the sum of all the free blocks, not the largest block that can be allocated. */
    size_t get_available_heapspace_in_bytes() { return m_free_bytes_remaining; }
    size_t get_minimum_ever_free_heap_size() { return m_minimum_ever_free_bytes_remaining; }
    /* The maximum size, in bytes, of all the free blocks within the heap at the time vPortGetHeapStats() is called. */
    size_t get_size_of_largest_free_block_in_bytes();  
    /* The minimum size, in bytes, of all the free blocks within the heap at the time vPortGetHeapStats() is called. */
    size_t get_size_of_smallest_free_block_in_bytes();
    /* The number of free memory blocks within the heap at the time vPortGetHeapStats() is called. */
    size_t get_number_of_free_blocks();
    void reset_heap_minimum_ever_free_heap_size();
    void heap_reset_state();
    
    void* mallocc(size_t size);
    void freee(void *data);
    
    void report();
};
typedef eosl_dev_memory_heap *eosl_dev_memory_heap_t;

#endif // GGML_EOSL_MEMORY_HPP
