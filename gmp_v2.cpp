#include <gmp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;
#define NUM_THEADS 8

static gmp_randstate_t gmp_randstate;
// 生成随机1024位大整数，每32位的高位为1
void generate_random_1024bit(mpz_t num, size_t bits) {
    mpz_urandomb(num, gmp_randstate, bits);
    // for (int i = 0; i < 32; i++) {
    //     mpz_setbit(num, i * 32 + 31);  // 每32位设置高位为1
    // }
}

// 测试函数
void benchmark(size_t size, size_t bits) {
    vector<mpz_t> a(size), b(size), e(size), res(size), r(size);
    for (size_t i = 0; i < size; i++) {
        mpz_init(a[i]);
        mpz_init(b[i]);
        mpz_init(e[i]);
        mpz_init(res[i]);
        mpz_init(r[i]);
        generate_random_1024bit(a[i], bits);
        generate_random_1024bit(b[i], bits);
        generate_random_1024bit(e[i], 512);
    }

    // 设置线程数
    omp_set_num_threads(NUM_THEADS);

    // 加法
    auto start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_add(res[i], a[i], b[i]);
    }
    auto end = high_resolution_clock::now();
    double ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Addition bits %d: %.f op/s \n", bits, ops);

    // 减法
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_sub(res[i], a[i], b[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Subtraction bits %d: %.f op/s \n", bits, ops);
    // 乘法
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_mul(res[i],a[i], b[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Multiplication bits %d: %.f op/s \n", bits, ops);

    // 除法
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_tdiv_qr(res[i], r[i], a[i], b[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Division bits %d: %.f op/s \n", bits, ops);

    // 模乘
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_mul(res[i], a[i], b[i]);
        mpz_mod(res[i], res[i], b[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Modular multiplication bits %d: %.f op/s \n", bits, ops);

    // 模幂
    int count = size;
    if (size >= 1024) size = 128 * 1024 / bits; 
    start = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_powm(res[i], a[i], b[i], b[i]);
    }
    end = high_resolution_clock::now();
    ops = size * 1000.0 * 1000.0 / duration_cast<microseconds>(end - start).count();
    printf("Modular exponentiation bits %d: %.f op/s \n", bits, ops);
    size = count;

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
    vector<size_t> bits_vec = {128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192};
    for (auto bits: bits_vec) {
        benchmark(400000, bits);
        cout << endl;
    }

    gmp_randclear(gmp_randstate);
    return 0;
}

