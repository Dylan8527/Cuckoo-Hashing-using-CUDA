#ifndef __CUDA_ARRAY_CUH__
#define __CUDA_ARRAY_CUH__
#include <cuda_runtime.h>
#include <cuda_utils.cuh>

template <typename T, uint S> // T is the type of the array, S is the size of the array 
struct HostArray; // array on host(cpu)

template <typename T, uint S> // T is the type of the array, S is the size of the array 
struct DeviceArray; // array on device(gpu)

//! Here we only consider memory allocation on array have the same type and the same size
template <typename T, uint S>       
struct HostArray {
    T* data;
    HostArray() : data(reinterpret_cast<T *>(malloc(sizeof(T) * S))) {}

    void free() { 
        if (data) {
            std::free(data);
            data = nullptr;
        }
    }

    // ~HostArray() { free(); }

    // copy data from host to host : host -> host
    HostArray &operator=(const HostArray<T, S> &rhs) {
        if (&rhs == this) return *this; // self-assignment
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyHostToHost)
        );
        return *this;
    }

    // copy data from device to host : device -> host
    HostArray &operator=(const DeviceArray<T, S> &rhs) {
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyDeviceToHost)
        );
        return *this;
    }

    T &operator()(uint i) { return data[i]; }
    const T &operator()(uint i) const { return data[i]; }

    [[nodiscard]] constexpr uint size() const { return S; }

};

// init array with the same value
template <typename T, uint S, T value>
__global__ void init_array_kernel(T *data) {
    cuda_foreach_uint(x, 0, S) {data[x] = value;}
}

template <typename T, uint S>
struct DeviceArray {
    T* data;
    DeviceArray() {
        CUDA_CHECK_ERROR(cudaMalloc(&data, sizeof(T) * S));
    }

    void free() {
        if (data) {
            CUDA_CHECK_ERROR(cudaFree(data));
            data = nullptr;
        }
    }

    // ~DeviceArray() { free(); }

    // copy data from host to device : host -> device
    DeviceArray &operator=(const HostArray<T, S> &rhs) {
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyHostToDevice)
        );
        return *this;
    }

    // copy data from device to device : device -> device
    DeviceArray &operator=(const DeviceArray<T, S> &rhs) {
        if (&rhs == this) return *this; // self-assignment
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyDeviceToDevice)
        );
        return *this;
    }

    [[nodiscard]] constexpr __host__ __device__ __forceinline__ uint 
    size() const { 
        return S; 
    }

    __device__ __forceinline__ T &operator()(uint i) { return data[i]; }
    __device__ __forceinline__ const T &operator()(uint i) const { return data[i]; }

    template<T value>
    void init() {
        init_array_kernel<T, S, value><<<ceil(S / static_cast<double>(BLOCKSIZE)), BLOCKSIZE>>>(data);
    }

};

#endif // __CUDA_ARRAY_CUH__