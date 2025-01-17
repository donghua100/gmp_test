#include <tommath.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;
#define NUM_THREADS 8

// 生成随机1024位大整数
void generate_random_1024bit(mp_int &num, int bits) {
    mp_rand(&num, bits / 8); // 128 bytes = 1024 bits
    // for (int i = 0; i < 128; i++) {
    //     num.dp[i] |= 0x80000000;  // 设置每32位的高位为1
    // }
}

// 测试函数
void benchmark(size_t size, int bits) {
    vector<mp_int> a(size), b(size), res(size), e(size), r(size);
    for (size_t i = 0; i < size; i++) {
        mp_init(&a[i]);
        mp_init(&b[i]);
        mp_init(&res[i]);
        mp_init(&e[i]);
        mp_init(&r[i]);
        generate_random_1024bit(a[i], bits);
        generate_random_1024bit(b[i], bits);
        generate_random_1024bit(e[i], 512);
    }
    omp_set_num_threads(NUM_THREADS);

    // 加法
    auto start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mp_add(&a[i], &b[i], &res[i]);
    }
    auto end = high_resolution_clock::now();
    double ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Addition bits %d: %.f op/s \n", bits, ops);

    // 减法
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mp_sub(&a[i], &b[i], &res[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Subtraction bits %d: %.f op/s \n", bits, ops);

    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mp_mul(&a[i], &b[i], &res[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Multiplication bits %d: %.f op/s \n", bits, ops);
    // 除法
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mp_div(&a[i], &b[i], &res[i], &r[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Division bits %d: %.f op/s \n", bits, ops);

    // 模乘
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mp_mul(&a[i], &b[i], &res[i]);
        mp_mod(&res[i], &b[i], &res[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Modular multiplication bits %d: %.f op/s \n", bits, ops);

    // 模幂
    int count = size;
    if (bits >= 1024) size = 500 * 8192 / bits; 
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mp_exptmod(&a[i], &r[i], &b[i], &res[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Modular exponentiation bits %d: %.f op/s \n", bits, ops);

    // 释放内存
    size = count;
    for (size_t i = 0; i < size; i++) {
        mp_clear(&a[i]);
        mp_clear(&b[i]);
        mp_clear(&res[i]);
    }
}

int main() {
    vector<int> bits_vec = {128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192};
    for (auto bits: bits_vec) {
        benchmark(10000, bits);
        cout << endl;
    }
    return 0;
}
//
//
//  g++ -fopenmp  tomath.cpp -o tomath -ltommath  
//   g++ -I/public/home/iscashpcb/libtommath -L /public/home/iscashpcb/libtommath/build -Wl,-rpath,/public/home/iscashpcb/libtommath/build -fopenmp  tomath.cpp -o tomath -ltommath

