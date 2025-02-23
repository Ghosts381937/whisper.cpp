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

#define ADDR_ALIGN(A, T, V) (T *)((((uint64_t)(A)) + (V - 1)) & (~(V - 1)))
#define ALIGN(N, T, V) ((((T)(N)) + (V - 1)) & (~(V - 1)))
 
#define READ(A, T)        *((volatile T *)((void *)(A)))
#define WRITE(A, T, V)    *((volatile T *)((void *)(A))) = (T)(V)

int c2h_devmem2(void *result, void *addr, char access_type);
int h2c_devmem2(void *result, void *addr, char access_type, uint64_t write_value);
