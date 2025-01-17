#include <gmp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

// 定义随机状态
static gmp_randstate_t gmp_randstate;

// 生成随机1024位大整数
void generate_random_1024bit(mpz_t &num) {
    mpz_rrandomb(num, gmp_randstate, 1024);
}

// 测试函数
void benchmark(size_t size) {
    vector<mpz_t> a(size), b(size), res(size);
    for (size_t i = 0; i < size; i++) {
        mpz_init(a[i]);
        mpz_init(b[i]);
        mpz_init(res[i]);
        generate_random_1024bit(a[i]);
        generate_random_1024bit(b[i]);
    }

    // 加法
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < size; i++) {
        mpz_add(res[i], a[i], b[i]);
    }
    auto end = high_resolution_clock::now();
    cout << "Addition time for size " << size << ": "
         << duration_cast<microseconds>(end - start).count() /1000.0 << " ms" << endl;

    // 减法
    start = high_resolution_clock::now();
    for (size_t i = 0; i < size; i++) {
        mpz_sub(res[i], a[i], b[i]);
    }
    end = high_resolution_clock::now();
    cout << "Subtraction time for size " << size << ": "
         << duration_cast<microseconds>(end - start).count() /1000.0 << " ms" << endl;

    // 除法
    start = high_resolution_clock::now();
    for (size_t i = 0; i < size; i++) {
        mpz_fdiv_q(res[i], a[i], b[i]);
    }
    end = high_resolution_clock::now();
    cout << "Division time for size " << size << ": "
         << duration_cast<microseconds>(end - start).count() /1000.0 << " ms" << endl;

    // 模乘
    start = high_resolution_clock::now();
    for (size_t i = 0; i < size; i++) {
        mpz_mul(res[i], a[i], b[i]);
        mpz_mod(res[i], res[i], b[i]);
    }
    end = high_resolution_clock::now();
    cout << "Modular multiplication time for size " << size << ": "
         << duration_cast<microseconds>(end - start).count() / 1000.0 << " ms" << endl;

    // 模幂
    start = high_resolution_clock::now();
    for (size_t i = 0; i < size; i++) {
        mpz_powm(res[i], a[i], b[i], b[i]);
    }
    end = high_resolution_clock::now();
    cout << "Modular exponentiation time for size " << size << ": "
         << duration_cast<microseconds>(end - start).count()/ 1000.0 << " ms" << endl;

    // 释放内存
    for (size_t i = 0; i < size; i++) {
        mpz_clear(a[i]);
        mpz_clear(b[i]);
        mpz_clear(res[i]);
    }
}

int main() {
    gmp_randinit_default(gmp_randstate);
    gmp_randseed_ui(gmp_randstate, time(NULL));

    for (size_t size = 100; size <= 3000; size += 100) {
        benchmark(size);
        cout << endl;
    }

    gmp_randclear(gmp_randstate);
    return 0;
}

