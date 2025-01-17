
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdio>

#define CUDA_CHECK(call)                                                          \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }


#define BLOCK_SIZE 16

__global__ void schoolbook_kernel(const uint32_t *a, const uint32_t *b, uint32_t *result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    printf("(%d, %d)\n", i, j);
    if (i < n && j < n) {
        result[i+j] = a[i] + b[j];
        // uint64_t temp = (uint64_t)a[i] * (uint64_t)b[j];
        // atomicAdd(&result[i + j], (uint32_t)(temp & 0xFFFFFFFF));  // 低 32 位
        // atomicAdd(&result[i + j + 1], (uint32_t)(temp >> 32));     // 高 32 位
    }
}
__global__ void printKernel() {
    printf("Hello from CUDA kernel!\n");
}

__global__ void comba_kernel(const uint32_t *a, const uint32_t *b, uint32_t *result, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;  // t 表示当前正在计算的结果索引

    if (t < 2 * n - 1) {
        uint64_t temp = 0;

        // 计算所有满足 i + j = t 的乘积累加
        for (int i = max(0, t - n + 1); i <= min(t, n - 1); i++) {
            int j = t - i;
            temp += (uint64_t)a[i] * (uint64_t)b[j];
        }

        // 存储结果的低 32 位和高 32 位（通过 atomicAdd 处理进位）
        atomicAdd(&result[t], (uint32_t)(temp & 0xFFFFFFFF));
        atomicAdd(&result[t + 1], (uint32_t)(temp >> 32));
    }
}


// 主函数：调用 CUDA 核函数
void multiply_cuda(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
                   std::vector<uint32_t> &result, int n, bool use_comba) {
    // 分配 GPU 内存
    uint32_t *d_a, *d_b, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, (2 * n) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_result, 0, (2 * n) * sizeof(uint32_t)));

    // 将数据从主机传输到设备
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 定义线程块和网格大小
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

    // 调用核函数
    if (use_comba) {
        int total_threads = 2 * n - 1;
        int threads_per_block = 256;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        comba_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_result, n);
    } else {
        schoolbook_kernel<<<blocks, threads>>>(d_a, d_b, d_result, n);
    }

    // 将结果传回主机
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, (2 * n) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 释放 GPU 内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void test() {
    int n = 1 << 20; // 假设向量大小为2^20
    size_t size = n * sizeof(float);
    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];

    // 分配设备内存
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // 初始化主机端数据
    for (int i = 0; i < n; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 检查结果
    for (int i = 0; i < n; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at %d., %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);
            exit(EXIT_FAILURE);
        }
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    printf("Test PASSED\n");
}

int main() {
    test();

    const int N = 32;  // 1024-bit 数字，由 32 个 32-bit 块组成

    printKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    // 输入大整数初始化
    std::vector<uint32_t> a(N, 0xFFFFFFFF);  // 示例：所有位都设置为 1
    std::vector<uint32_t> b(N, 0xFFFFFFFF);
    std::vector<uint32_t> result(2 * N, 0);

    // 执行 Schoolbook 乘法
    std::cout << "Schoolbook 乘法结果:" << std::endl;
    multiply_cuda(a, b, result, N, false);
    for (int i = 2 * N - 1; i >= 0; i--) {
        printf("%08X", result[i]);
    }
    std::cout << std::endl;

    // 清空结果
    std::fill(result.begin(), result.end(), 0);

    // 执行 Comba 乘法
    std::cout << "Comba 乘法结果:" << std::endl;
    multiply_cuda(a, b, result, N, true);
    for (int i = 2 * N - 1; i >= 0; i--) {
        printf("%08X", result[i]);
    }
    std::cout << std::endl;

    return 0;
}

