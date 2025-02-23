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

#include <string>
#include <mutex>
#include <vector>
		
#include "ggml-eosl_host.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-eosl_host/host_system.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wimport-preprocessor-directive-pedantic"
#import "ggml-eosl_host/host_debug.hpp"
#pragma clang diagnostic pop
#else
#include "ggml-eosl_host/host_debug.hpp"
#endif

#define TAG "EOSL"
#define UNUSED GGML_UNUSED

struct ggml_backend_eosl_host_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static char *dump_tensor(const ggml_tensor *tensor)    {
    static char buffer[256];

    snprintf(buffer, sizeof(buffer), "[tensor(%s, %p): buffer(%p), data(%p), data_size(0X%" PRIX64 ")]",
             (tensor->name[0] == '\0') ? "N/A" : tensor->name,
             (void *)tensor, (void *)(tensor->buffer), tensor->data, tensor->data_size);
    return buffer;
}

static char *dump_cgraph(const ggml_cgraph *cgraph)    {
    static char buffer[2048];
    int index = 0, i;
    
    index += snprintf(buffer + index, sizeof(buffer) - index,
                      "**** ggml_cgraph(%p) info ****\n", (void *)cgraph);
    index += snprintf(buffer + index, sizeof(buffer) - index,
                      "  n_nodes(%d), n_leafs(%d), order(%d)\n",
                      cgraph->n_nodes, cgraph->n_leafs, cgraph->order);
    for(i = 0; i < cgraph->n_nodes; i++)
        index += snprintf(buffer + index, sizeof(buffer) - index,
                          "    node[%d]: %s\n",
                          i, dump_tensor(cgraph->nodes[i]));
    buffer[index++] = '\n';
    for(i = 0; i < cgraph->n_leafs; i++)
        index += snprintf(buffer + index, sizeof(buffer) - index,
                          "    leaf[%d]: %s\n",
                          i, dump_tensor(cgraph->leafs[i]));
    
    return buffer;
}

struct ggml_backend_eosl_host_buffer_type_context {
    int device;
    std::string name;
    size_t alignment;
    size_t max_size;

    explicit ggml_backend_eosl_host_buffer_type_context(int device, std::string name, size_t align, size_t max):
        device(device), name(name), alignment(align), max_size(max)    {
    }
};

struct ggml_backend_eosl_host_context {
    int device;

    explicit ggml_backend_eosl_host_context(int device): device(device)    {
    }
};

struct ggml_backend_eosl_host_buffer_context {
    int device;
    void *device_ptr = nullptr;
    size_t size;

    explicit ggml_backend_eosl_host_buffer_context(int device, void *device_ptr, size_t size):
        device(device), device_ptr(device_ptr), size(size)    {
    }
};

static void ggml_eosl_serialize_graph(const ggml_cgraph *cgraph, std::vector<uint8_t> &output) {

    //ggml_graph_print(cgraph); // TODO: segmentation fault
  
    // serialization format:
    //     n_nodes (4 bytes) | grads ? (4 bytes) | n_leafs (4 bytes) | 
    //     nodes (n_nodes * sizeof(uint64_t) | tensors (n_leafs * sizeof(uint64_t)) |
    int32_t n_nodes = cgraph->n_nodes;
    int32_t n_leafs = cgraph->n_leafs;
    int output_size = sizeof(uint32_t) +
                      sizeof(uint32_t) + n_nodes * sizeof(uint64_t) +
                      sizeof(uint32_t) + n_leafs * sizeof(uint64_t);
    output.resize(output_size, 0);
    uint8_t *buffer = output.data();
    *((int32_t *)buffer) = n_nodes;
    buffer += sizeof(int32_t);
    *((int32_t *)buffer) = (cgraph->grads == nullptr) ? 0 : 1;
    buffer += sizeof(int32_t);
    *((int32_t *)buffer) = n_leafs;
    buffer += sizeof(int32_t);
    memcpy((void *)buffer, (void *)(cgraph->nodes), n_nodes * sizeof(uint64_t));
    buffer += n_nodes * sizeof(uint64_t);
    memcpy((void *)buffer, (void *)(cgraph->leafs), n_leafs * sizeof(uint64_t));
}

static void ggml_eosl_serialize_tensor(ggml_eosl_serialized_tensor *serialized_tensor,
                                       const ggml_tensor *tensor)    {
    serialized_tensor->id = reinterpret_cast<uint64_t>(tensor);
    serialized_tensor->type = tensor->type;
    serialized_tensor->buffer_id = reinterpret_cast<uint64_t>(tensor->buffer); // buffer id
    
    for(uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        serialized_tensor->ne[i] = tensor->ne[i];
        serialized_tensor->nb[i] = tensor->nb[i];
    }
    serialized_tensor->op = tensor->op;
    for(uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        serialized_tensor->op_params[i] = tensor->op_params[i];
    }
    serialized_tensor->flags = tensor->flags;
    for(uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        serialized_tensor->src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    serialized_tensor->view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    serialized_tensor->view_offs = tensor->view_offs;
    serialized_tensor->data = reinterpret_cast<uint64_t>(tensor->data);
    serialized_tensor->data_size = reinterpret_cast<uint64_t>(tensor->data_size);
    snprintf(serialized_tensor->name, GGML_MAX_NAME, "%s", tensor->name);
    
    EOSL_LOG_DEBUG_S(TAG, "%s: data(0x%" PRIx64 ")", __func__, serialized_tensor->data);
}

#define GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(BUFFER) ({                    \
    ggml_backend_eosl_host_buffer_type_context *buft_ctx =                          \
        (ggml_backend_eosl_host_buffer_type_context *)(BUFFER->buft->context);      \
    ggml_eosl_host_backend_dev *device =                                            \
        ggml_eosl_host::get_ggml_eosl_host()->get_backend_device(buft_ctx->device); \
    device;                                                                         \
})

static void ggml_backend_eosl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_eosl_host_backend_dev *host_backend_dev = GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(buffer);
    ggml_backend_eosl_host_buffer_context *ctx = (ggml_backend_eosl_host_buffer_context *)buffer->context;
    if(ctx->device_ptr != nullptr)    {
        host_backend_dev->free_memory(ctx->device_ptr);
    }

    EOSL_LOG_DEBUG_S(TAG, "%s: %p", __func__, (void *)buffer);

    host_backend_dev->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_FREE_BUFFER,
                                        (uint64_t)buffer, (uint64_t)ctx->device_ptr, (uint64_t)ctx->size,
                                        (uint64_t)nullptr, 0);

    delete ctx;
}

static void *ggml_backend_eosl_host_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_eosl_host_buffer_context *ctx = (ggml_backend_eosl_host_buffer_context *)buffer->context;
    EOSL_LOG_DEBUG_S(TAG, "%s: %p/%p", __func__, (void *)buffer, ctx->device_ptr);

    return ctx->device_ptr;
}

/* Copy from ggml_backend_cpu_aarch64_buffer_init_tensor */
static void ggml_backend_eosl_host_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor) {
    ggml_eosl_serialized_tensor s_tensor;
    ggml_eosl_host_backend_dev *device = GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(buffer);

    if(tensor->view_src != NULL)    { // ggml_is_view
        tensor->data_size = tensor->view_src->data_size;
    } else    {
        tensor->data_size = ggml_backend_buffer_get_alloc_size(buffer, tensor);
    }

    EOSL_LOG_DEBUG_S(TAG, "%s: buffer(%p) : %s", __func__, (void *)buffer, dump_tensor(tensor));

    //tensor->extra = (void *)ggml_aarch64_get_optimal_repack_type(tensor); // NOLINT
    ggml_eosl_serialize_tensor(&s_tensor, tensor);
    device->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_INIT_TENSOR,
                              (uint64_t)&s_tensor, (uint64_t)sizeof(s_tensor),
                              (uint64_t)nullptr, 0);
}

static void ggml_backend_eosl_host_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor,
                                                     const void *data, size_t offset, size_t size) {
    ggml_eosl_host_backend_dev *device = GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(buffer);

#if 1
    EOSL_LOG_DEBUG_S(TAG, "%s: buffer(0x%p) : %s, data(%p)/offset(0x%zx)/size(0x%zx)",
                     __func__, (void *)buffer, dump_tensor(tensor), data, offset, size);
#else
    EOSL_LOG_DEBUG_S(TAG, "%s: buffer(0x%p) : %s, data(%p)/offset(0x%zx)/size(0x%zx)\n"
                     "        0x%" PRIx32 ", 0x%" PRIx32 ", 0x%" PRIx32 ", 0x%" PRIx32 
                     ", 0x%" PRIx32 ", 0x%" PRIx32 ", 0x%" PRIx32 ", 0x%" PRIx32,
                     __func__, (void *)buffer, dump_tensor(tensor), data, offset, size,
                     *((uint32_t *)(((uint8_t *)(data)) + 0)), *((uint32_t *)(((uint8_t *)(data)) + 4)),
                     *((uint32_t *)(((uint8_t *)(data)) + 8)), *((uint32_t *)(((uint8_t *)(data)) + 12)),
                     *((uint32_t *)(((uint8_t *)(data)) + 16)), *((uint32_t *)(((uint8_t *)(data)) + 20)),
                     *((uint32_t *)(((uint8_t *)(data)) + 24)), *((uint32_t *)(((uint8_t *)(data)) + 28)));
#endif
        
    device->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_SET_TENSOR,
                              (uint64_t)buffer,
                              (uint64_t)tensor, (uint64_t)(tensor->data), (uint64_t)data, (uint64_t)offset, (uint64_t)size,
                              (uint64_t)nullptr, 0);
}

static void ggml_backend_eosl_host_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor *tensor,
                                                     void *data, size_t offset, size_t size) {
    ggml_eosl_host_backend_dev *device = GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(buffer);
    EOSL_LOG_DEBUG_S(TAG, "%s: %p : %s, %p/0x%zx/0x%zx", __func__, (void *)buffer, dump_tensor(tensor), data, offset, size);
    device->dma_from_device(tensor->data, ((uint8_t *)data) + offset, 0, size, 0);
}

static bool ggml_backend_eosl_host_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                     const ggml_tensor *src, ggml_tensor *dst) {
    UNUSED(buffer); UNUSED(src); UNUSED(dst);
    EOSL_LOG_DEBUG_S(TAG, "%s: %p : %s, %s", __func__, (void *)buffer, dump_tensor(src), dump_tensor(dst));
    ggml_eosl_host_backend_dev *device = GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(buffer);

    device->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_COPY_TENSOR,
                              (uint64_t)buffer, (uint64_t)src, (uint64_t)dst,
                              (uint64_t)nullptr, 0);
    return true;
}

static void ggml_backend_eosl_host_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                        struct ggml_tensor *tensor,
                                                        uint8_t value, size_t offset,
                                                        size_t size)    {
    EOSL_LOG_DEBUG_S(TAG, "%s: buffer(%p), tenser(%p)", __func__, (void *)buffer, (void *)tensor);
    ggml_eosl_host_backend_dev *device = GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(buffer);

    device->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_MEMSET_TENSOR,
                              (uint64_t)buffer, (uint64_t)tensor,
                              (uint64_t)value, (uint64_t)offset, (uint64_t)size,
                              (uint64_t)nullptr, 0);
}

static void ggml_backend_eosl_host_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    EOSL_LOG_DEBUG_S(TAG, "%s: %p", __func__, (void *)buffer);
    
    ggml_eosl_host_backend_dev *device = GET_BACKEND_DEVICE_FROM_GGML_BACKEND_BUFFER_T(buffer);
    ggml_backend_eosl_host_buffer_context *ctx = (ggml_backend_eosl_host_buffer_context *)buffer->context;
    device->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_BUFFER_CLEAR,
                              (uint64_t)ctx->device_ptr, (uint64_t)(buffer->size), (uint64_t)value,
                              (uint64_t)nullptr, 0);
}

static ggml_backend_buffer_i ggml_backend_eosl_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_eosl_host_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_eosl_host_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_eosl_host_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_eosl_host_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_eosl_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_eosl_host_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_eosl_host_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_eosl_host_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_eosl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_eosl_host_buffer_type_context *buft_ctx = (ggml_backend_eosl_host_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_eosl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_eosl_host_buffer_type_context *buft_ctx = (ggml_backend_eosl_host_buffer_type_context *)buft->context;
    ggml_eosl_host_backend_dev *host_backend_dev = ggml_eosl_host::get_ggml_eosl_host()->get_backend_device(buft_ctx->device);
    void *device_ptr = host_backend_dev->malloc_memory(size);
    if(device_ptr != nullptr) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_eosl_host_buffer_interface,
            new ggml_backend_eosl_host_buffer_context{buft_ctx->device, device_ptr, size},
            size);

        EOSL_LOG_DEBUG_S(TAG, "%s: %p/%p/0x%zx", __func__, (void *)buffer, device_ptr, size);
        
        host_backend_dev->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_ALLOC_BUFFER,
                                            (uint64_t)buffer, (uint64_t)device_ptr, (uint64_t)size,
                                            (uint64_t)nullptr, 0);

        return buffer;
    } else {
        return nullptr;
    }
}

static size_t ggml_backend_eosl_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_eosl_host_buffer_type_context *buft_ctx = (ggml_backend_eosl_host_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t ggml_backend_eosl_host_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_eosl_host_buffer_type_context *buft_ctx = (ggml_backend_eosl_host_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_eosl_host_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor *tensor) {
    UNUSED(buft);

    size_t size = ggml_nbytes(tensor);
    EOSL_LOG_DEBUG_S(TAG, "%s:%s:%p: 0x%zx", __func__, tensor->name, (void *)tensor, size);
    return size;
}

static ggml_backend_buffer_type_i ggml_backend_eosl_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_eosl_host_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_eosl_host_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_eosl_host_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_eosl_host_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_eosl_host_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_eosl_host_name(ggml_backend_t backend) {
    ggml_backend_eosl_host_context *eosl_ctx = (ggml_backend_eosl_host_context *)backend->context;
    ggml_eosl_host_backend_dev *backend_device = 
        ggml_eosl_host::get_ggml_eosl_host()->get_backend_device(eosl_ctx->device);

    return backend_device->get_name();
}

static void ggml_backend_eosl_host_free(ggml_backend_t backend) {
    ggml_backend_eosl_host_context *eosl_ctx = (ggml_backend_eosl_host_context *)backend->context;
    delete eosl_ctx;
    delete backend;

    EOSL_LOG_DEBUG_S(TAG, "%s", __func__);
}

static void ggml_backend_eosl_host_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
    // this is no-op because we don't have any async operations

    EOSL_LOG_DEBUG_S(TAG, "%s", __func__);
}

static enum ggml_status ggml_backend_eosl_host_graph_compute(ggml_backend_t backend, ggml_cgraph *cgraph) {
    EOSL_LOG_DEBUG_S(TAG, "%s", __func__);
    ggml_backend_eosl_host_context *eosl_ctx = (ggml_backend_eosl_host_context *)backend->context;
    ggml_eosl_host_backend_dev *backend_device = 
        ggml_eosl_host::get_ggml_eosl_host()->get_backend_device(eosl_ctx->device);
    std::vector<uint8_t> input;
    ggml_eosl_serialize_graph(cgraph, input);
    
    backend_device->send_eosl_dev_cmd(EOSL_GGML_DEV_CMD_GRAPH_COMPUTE,
                                      (uint64_t)(input.data()), (uint64_t)(input.size()),
                                      (uint64_t)nullptr, 0);

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_eosl_host_interface = {
    /* .get_name                = */ ggml_backend_eosl_host_name,
    /* .free                    = */ ggml_backend_eosl_host_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_eosl_host_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_eosl_host_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

GGML_API ggml_backend_buffer_type_t ggml_backend_eosl_host_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    EOSL_LOG_DEBUG_S(TAG, "call %s on device %d", __func__, device);

    int dev_count = ggml_backend_eosl_host_get_device_count();
    if(device >= dev_count || device < 0) {
        EOSL_LOG_ERROR(TAG, "device_index: %d is out of range [0, %d]", device, dev_count - 1);
        GGML_ASSERT(device < dev_count);
    }

    static struct ggml_backend_buffer_type ggml_backend_eosl_host_buffer_types[GGML_EOSL_MAX_DEVICES];
    static bool ggml_backend_eosl_host_buffer_type_initialized = false;

    if(!ggml_backend_eosl_host_buffer_type_initialized) {
        for (int i = 0; i < dev_count; i++) {
            size_t alignment = (size_t)BYTE_ALIGNMENT;
            size_t max_size = (size_t)0X10000000;  // TODO: request from device
            ggml_backend_eosl_host_buffer_type_context *buft_ctx = new ggml_backend_eosl_host_buffer_type_context {
                /* .device  = */ i,
                /* .name      = */ GGML_EOSL_NAME "-" + std::to_string(i),
                /* .alignment = */ alignment,
                /* .max_size  = */ max_size
            };

            ggml_backend_eosl_host_buffer_types[i] = {
                /* .iface    = */ ggml_backend_eosl_host_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_eosl_host_reg(), i),
                /* .context  = */ buft_ctx,
            };
        }

        ggml_backend_eosl_host_buffer_type_initialized = true;
    }

    return &ggml_backend_eosl_host_buffer_types[device];
}

ggml_backend_t ggml_backend_eosl_host_init(int device) {
    ggml_eosl_host *host = ggml_eosl_host::get_ggml_eosl_host();
    
    host->check_allow_gpu_index(device);

    ggml_backend_eosl_host_context *ctx = new ggml_backend_eosl_host_context(device);
    if (ctx == nullptr) {
        EOSL_LOG_ERROR(TAG, "Failed to allocate context");
        return nullptr;
    };

    ggml_backend_t eosl_backend = new ggml_backend {
        /* .guid      = */ ggml_eosl_get_backend_guid(),
        /* .interface = */ ggml_backend_eosl_host_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_eosl_host_reg(), device),
        /* .context   = */ ctx
    };

    return eosl_backend;
}

bool ggml_backend_is_eosl_host(ggml_backend_t backend) {
    return (backend != NULL) && ggml_guid_matches(backend->guid, ggml_eosl_get_backend_guid());
}

int ggml_backend_eosl_host_get_device_count() {
    return ggml_eosl_host::get_ggml_eosl_host()->get_device_count();
}

/**** device interface ****/

struct ggml_backend_eosl_host_device_context {
   int device;
   std::string name;
   std::string description;
};

void ggml_backend_eosl_host_get_device_memory(const int device, size_t *free, size_t *total) {
    ggml_eosl_host_backend_dev *backend_device = ggml_eosl_host::get_ggml_eosl_host()->get_backend_device(device);
    backend_device->get_device_memory(free, total);
}

static const char *ggml_backend_eosl_host_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_eosl_host_device_context * ctx = (ggml_backend_eosl_host_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char *ggml_backend_eosl_host_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_eosl_host_device_context * ctx = (ggml_backend_eosl_host_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_eosl_host_device_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    ggml_backend_buffer_type_t buft = dev->iface.get_buffer_type(dev);
    ggml_backend_eosl_host_buffer_type_context *buft_ctx = (ggml_backend_eosl_host_buffer_type_context *)buft->context;
    ggml_eosl_host_backend_dev *backend_device = ggml_eosl_host::get_ggml_eosl_host()->get_backend_device(buft_ctx->device);
    backend_device->get_device_memory(free, total);
}

static enum ggml_backend_dev_type ggml_backend_eosl_host_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);

    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_eosl_host_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_eosl_host_device_get_name(dev);
    props->description = ggml_backend_eosl_host_device_get_description(dev);
    props->type        = ggml_backend_eosl_host_device_get_type(dev);
    ggml_backend_eosl_host_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_eosl_host_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    ggml_backend_eosl_host_device_context * ctx = (ggml_backend_eosl_host_device_context *)dev->context;

    return ggml_backend_eosl_host_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_eosl_host_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_eosl_host_device_context *ctx = (ggml_backend_eosl_host_device_context *)dev->context;

    return ggml_backend_eosl_host_buffer_type(ctx->device);
}

static bool ggml_backend_eosl_host_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);
    
    EOSL_LOG_DEBUG_S(TAG, "%s: %s", __func__, op->name);

    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_eosl_host_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_eosl_host_buffer_type_name) {
        return false;
    }

    ggml_backend_eosl_host_buffer_type_context *buft_ctx = (ggml_backend_eosl_host_buffer_type_context *)buft->context;
    ggml_backend_eosl_host_device_context *dev_ctx = (ggml_backend_eosl_host_device_context *)dev->context;
    return buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_eosl_host_device_interface = {
    /* .get_name             = */ ggml_backend_eosl_host_device_get_name,
    /* .get_description      = */ ggml_backend_eosl_host_device_get_description,
    /* .get_memory           = */ ggml_backend_eosl_host_device_get_memory,
    /* .get_type             = */ ggml_backend_eosl_host_device_get_type,
    /* .get_props            = */ ggml_backend_eosl_host_device_get_props,
    /* .init_backend         = */ ggml_backend_eosl_host_device_init,
    /* .get_buffer_type      = */ ggml_backend_eosl_host_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_eosl_host_device_supports_op,
    /* .supports_buft        = */ ggml_backend_eosl_host_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_eosl_host_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_EOSL_NAME;
}

static size_t ggml_backend_eosl_host_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_eosl_host_reg_context *ctx = (ggml_backend_eosl_host_reg_context *)reg->context;

    return ctx->devices.size();
}


static ggml_backend_dev_t ggml_backend_eosl_host_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_eosl_host_reg_context *ctx = (ggml_backend_eosl_host_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());

    return ctx->devices[index];
}

static void * ggml_backend_eosl_host_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg); GGML_UNUSED(name);

    // TODO: update to the current function signature
    //if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
    //    return (void *)ggml_backend_sycl_split_buffer_type;
    //}  

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_eosl_host_reg_interface = {
    /* .get_name         = */ ggml_backend_eosl_host_reg_get_name,
    /* .get_device_count = */ ggml_backend_eosl_host_reg_get_device_count,
    /* .get_device       = */ ggml_backend_eosl_host_reg_get_device,
    /* .get_proc_address = */ ggml_backend_eosl_host_get_proc_address,
};

ggml_backend_reg_t ggml_backend_eosl_host_reg(void) {
    static ggml_backend_reg ggml_backend_eosl_host_reg;
    ggml_backend_eosl_host_device_context *dev_ctx;
    static bool initialized = false;
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_eosl_host_reg_context *ctx = new ggml_backend_eosl_host_reg_context;

            for (int i = 0; i < ggml_eosl_host::get_ggml_eosl_host()->get_device_count(); i++) {
                dev_ctx = new ggml_backend_eosl_host_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_EOSL_NAME "-" + std::to_string(i);
                dev_ctx->description = "VCU118 (MCU * 1, Andes NX27V * 2)";

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .interface = */ ggml_backend_eosl_host_device_interface,
                    /* .reg       = */ &ggml_backend_eosl_host_reg,
                    /* .context   = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            ggml_backend_eosl_host_reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .interface =   */ ggml_backend_eosl_host_reg_interface,
                /* .context   =   */ ctx
            };
        }

        initialized = true;
    }

    return &ggml_backend_eosl_host_reg;
}
