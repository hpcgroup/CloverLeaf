/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "shared.h"

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include "umpire/TypedAllocator.hpp"
#include "RAJA/RAJA.hpp"

#define CLOVER_DEFAULT_BLOCK_SIZE (256)
#define DEVICE_KERNEL __host__ __device__

// #define CLOVER_SYNC_ALL_KERNELS
// #define CLOVER_MANAGED_ALLOC

#ifdef CLOVER_MANAGED_ALLOC
  #define CLOVER_MEMCPY_KIND_D2H (cudaMemcpyDefault)
  #define CLOVER_MEMCPY_KIND_H2D (cudaMemcpyDefault)
#else
  #define CLOVER_MEMCPY_KIND_D2H (cudaMemcpyDeviceToHost)
  #define CLOVER_MEMCPY_KIND_H2D (cudaMemcpyHostToDevice)
#endif

#ifdef CLOVER_SYNC_ALL_KERNELS
  #define CLOVER_BUILTIN_FILE __builtin_FILE()
  #define CLOVER_BUILTIN_LINE __builtin_LINE()
#else
  #define CLOVER_BUILTIN_FILE ("")
  #define CLOVER_BUILTIN_LINE (0)
#endif

#define RAJA_BLOCK_SIZE 256

namespace clover {

struct CLResourceManager{
  typedef enum {
    UNKNOWN = -1,
    HOST = 0,
    DEVICE = 1,
    UM = 2,
    RSEND
  } UmpireResourceType;

  static std::string umpireResources[UmpireResourceType::RSEND];
  static std::string CloverLeafPools[UmpireResourceType::RSEND];
  static int allocator_ids[UmpireResourceType::RSEND];
};

struct chunk_context {};
struct context {};

template <typename T> static inline T *alloc(size_t count) {
  static auto& rm = umpire::ResourceManager::getInstance();
#ifdef CLOVER_MANAGED_ALLOC
  auto alloc = rm.getAllocator(CLResourceManager::allocator_ids[CLResourceManager::UM]);
#else
  auto alloc = rm.getAllocator(CLResourceManager::allocator_ids[CLResourceManager::DEVICE]);
#endif

  T* p = static_cast<T*>(alloc.allocate(count * sizeof(T)));

  if ( p == nullptr) {
    std::cerr << "Failed to allocate memory of " << count << "bytes:";
    std::abort();
  }
  return p;
}

static inline void dealloc(void *p) {
  static auto& rm = umpire::ResourceManager::getInstance();
  rm.getAllocator(p).deallocate(p);
}

template <typename T> struct Buffer1D {
  size_t size;
  T *data;
  Buffer1D(context &, size_t size) : size(size), data(alloc<T>(size)) {}
  Buffer1D(context &, size_t size, T *host_init) : size(size), data(alloc<T>(size)) {
    static auto& rm = umpire::ResourceManager::getInstance();
    rm.copy(data, host_init, size * sizeof(T));
  }
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  Buffer1D(Buffer1D &&other) noexcept : size(other.size), data(std::exchange(other.data, nullptr)) {}
  //  Buffer1D(const Buffer1D<T> &that) : size(that.size), data(that.data) {}
  // XXX this model captures this class by value, the dtor is called for each lambda scope which is wrong!
  //  ~Buffer1D() { std::free(data); }

  void release() { dealloc(data); }

  __host__ __device__ T &operator[](size_t i) const { return data[i]; }
  T *actual() { return data; }

  template <size_t D> [[nodiscard]] size_t extent() const {
    static_assert(D < 1);
    return size;
  }

  std::vector<T, umpire::TypedAllocator<T>> mirrored() const {
    static auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator alloc = rm.getAllocator(CLResourceManager::allocator_ids[CLResourceManager::HOST]);
    umpire::TypedAllocator<T> vector_allocator(alloc);

    std::vector<T, umpire::TypedAllocator<T>> buffer(size, vector_allocator);
    rm.copy(buffer.data(), data, size * sizeof(T));
    return buffer;
  }
};

template <typename T> struct Buffer2D {
  size_t sizeX, sizeY;
  T *data;
  Buffer2D(context &, size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), data(alloc<T>(sizeX * sizeY)) {}
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  Buffer2D(Buffer2D &&other) noexcept : sizeX(other.sizeX), sizeY(other.sizeY), data(std::exchange(other.data, nullptr)) {}
  //  Buffer2D(const Buffer2D<T> &that) : sizeX(that.sizeX), sizeY(that.sizeY), data(that.data) {}
  // XXX this model captures this class by value, the dtor is called for each lambda scope which is wrong!
  // ~Buffer2D() { std::free(data); }

  void release() { dealloc(data); }

  __host__ __device__ T &operator()(size_t i, size_t j) const { return data[i + j * sizeX]; }
  T *actual() { return data; }

  template <size_t D> [[nodiscard]] size_t extent() const {
    if constexpr (D == 0) {
      return sizeX;
    } else if (D == 1) {
      return sizeY;
    } else {
      static_assert(D < 2);
    }
  }

  std::vector<T, umpire::TypedAllocator<T>> mirrored() const {
    static auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator alloc = rm.getAllocator(CLResourceManager::allocator_ids[CLResourceManager::HOST]);
    umpire::TypedAllocator<T> vector_allocator(alloc);

    std::vector<T, umpire::TypedAllocator<T>> buffer(sizeX * sizeY, vector_allocator);
    rm.copy(buffer.data(), data, buffer.size() * sizeof(T));
    return buffer;
  }

  clover::BufferMirror2D<T> mirrored2() {
    return {mirrored().data(), extent<0>(), extent<1>()}; }
};

template <typename T> using StagingBuffer1D = T*;

} // namespace clover

using clover::Range1d;
using clover::Range2d;

template<typename T>
void raja_copy(T* dest, T* src, size_t bytes){
  static auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(dest, src, bytes);
}


#ifdef __ENABLE_CUDA__
using raja_default_policy= RAJA::cuda_exec<RAJA_BLOCK_SIZE>;
using reduce_policy = RAJA::cuda_reduce;
using rajaDeviceProp = cudaDeviceProp;
using rajaError_t = cudaError_t;

using KERNEL_EXEC_POL_CUDA  = RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<8>, RAJA::cuda_block_y_direct,
          RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::cuda_block_x_direct,
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;

namespace clover{
static inline void checkError(const cudaError_t err = cudaGetLastError()) {
  if (err != cudaSuccess) {
    std::cerr << std::string(cudaGetErrorName(err)) + ": " + std::string(cudaGetErrorString(err)) << std::endl;
    std::abort();
  }
}
}

static inline rajaError_t rajaDeviceSynchronize() {
  return cudaDeviceSynchronize();
}

static inline rajaError_t rajaGetDeviceProperties(rajaDeviceProp * props, int id){
  return cudaGetDeviceProperties(props, id);
}

static inline rajaError_t rajaSetDevice(int id){
  return cudaSetDevice(id);
}

static inline rajaError_t rajaGetDevice(int* id){
  return cudaGetDevice(id);
}

static inline rajaError_t rajaGetDeviceCount(int *count){
  return cudaGetDeviceCount(count);
}

#elif __ENABLE_HIP__
using raja_default_policy= RAJA::hip_exec<RAJA_BLOCK_SIZE>;
using reduce_policy = RAJA::hip_reduce;
using rajaDeviceProp = hipDeviceProp_t;
using rajaError_t = hipError_t;

using KERNEL_EXEC_POL_CUDA  = RAJA::KernelPolicy<
    RAJA::statement::HipKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<8>, RAJA::hip_block_y_direct,
          RAJA::statement::Tile<0, RAJA::tile_fixed<32>, RAJA::hip_block_x_direct,
            RAJA::statement::For<1, RAJA::hip_thread_y_direct,
              RAJA::statement::For<0, RAJA::hip_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;

namespace clover{
static inline void checkError(const hipError_t err = hipGetLastError()) {
  if (err != hipSuccess) {
    std::cerr << std::string(hipGetErrorName(err)) + ": " + std::string(hipGetErrorString(err)) << std::endl;
    std::abort();
  }
}
}

static inline rajaError_t rajaDeviceSynchronize() {
  return hipDeviceSynchronize();
}

static inline rajaError_t rajaGetDeviceProperties(rajaDeviceProp * props, int id){
  return hipGetDeviceProperties(props, id);
}

static inline rajaError_t rajaSetDevice(int id){
  return hipSetDevice(id);
}

static inline rajaError_t rajaGetDevice(int* id){
  return hipGetDevice(id);
}

static inline rajaError_t rajaGetDeviceCount(int *count){
  return hipGetDeviceCount(count);
}

#endif

