#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256  // 每个线程块的线程数

// 将 64 位结果拆分成两个 32 位数
__device__ void split_64bit(uint64_t value, uint32_t &low, uint32_t &high) {
    low = static_cast<uint32_t>(value & 0xFFFFFFFF);
    high = static_cast<uint32_t>(value >> 32);
}

// 核函数：并行计算小块乘法
__global__ void karatsuba_kernel(const uint32_t *a, const uint32_t *b, uint32_t *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        uint64_t temp = static_cast<uint64_t>(a[idx]) * static_cast<uint64_t>(b[idx]);
        uint32_t low, high;
        split_64bit(temp, low, high);

        // 原子操作累加结果，避免线程冲突
        atomicAdd(&result[2 * idx], low);
        atomicAdd(&result[2 * idx + 1], high);
    }
}

// 并行加法核函数
__global__ void add_kernel(uint32_t *result, const uint32_t *z0, const uint32_t *z1, const uint32_t *z2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(&result[idx], z0[idx]);   // 累加 z0
        atomicAdd(&result[idx + n / 2], z1[idx]);  // 累加 z1
        atomicAdd(&result[idx + n], z2[idx]);  // 累加 z2
    }
}

// 主 Karatsuba 乘法函数（CPU 控制）
void karatsuba_cuda(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b, std::vector<uint32_t> &result) {
    int n = a.size();

    // 设备内存分配
    uint32_t *d_a, *d_b, *d_z0, *d_z1, *d_z2, *d_result;
    cudaMalloc(&d_a, n * sizeof(uint32_t));
    cudaMalloc(&d_b, n * sizeof(uint32_t));
    cudaMalloc(&d_result, 2 * n * sizeof(uint32_t));
    cudaMalloc(&d_z0, n * sizeof(uint32_t));
    cudaMalloc(&d_z1, n * sizeof(uint32_t));
    cudaMalloc(&d_z2, n * sizeof(uint32_t));

    // 数据传输到 GPU
    cudaMemcpy(d_a, a.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, 2 * n * sizeof(uint32_t));

    // 线程配置
    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;

    // 计算 z0, z1, z2
    karatsuba_kernel<<<blocks, threads>>>(d_a, d_b, d_z0, n);
    karatsuba_kernel<<<blocks, threads>>>(d_a, d_b, d_z1, n / 2);
    karatsuba_kernel<<<blocks, threads>>>(d_a + n / 2, d_b + n / 2, d_z2, n / 2);
    cudaDeviceSynchronize();

    // 合并结果
    add_kernel<<<blocks, threads>>>(d_result, d_z0, d_z1, d_z2, n);
    cudaDeviceSynchronize();

    // 结果传回 CPU
    cudaMemcpy(result.data(), d_result, 2 * n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_z0);
    cudaFree(d_z1);
    cudaFree(d_z2);
    cudaFree(d_result);
}

int main() {
    // 定义 1024-bit 数字（32 个 32-bit 块）
    const int N = 32;
    std::vector<uint32_t> a(N, 0xFFFFFFFF);  // 示例：1024-bit 最大值
    std::vector<uint32_t> b(N, 0xFFFFFFFF);
    std::vector<uint32_t> result(2 * N, 0);

    // 调用 CUDA Karatsuba 乘法
    karatsuba_cuda(a, b, result);

    // 打印结果
    std::cout << "Result (2048-bit): ";
    for (int i = 2 * N - 1; i >= 0; --i) {
        printf("%08X", result[i]);
    }
    std::cout << std::endl;

    return 0;
}

