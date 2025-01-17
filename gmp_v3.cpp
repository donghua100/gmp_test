#include <gmp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;
#define NUM_THEADS 40

static gmp_randstate_t gmp_randstate;
// 生成随机1024位大整数，每32位的高位为1
void generate_random_1024bit(mpz_t num, size_t bits) {
    mpz_urandomb(num, gmp_randstate, bits);
    // for (int i = 0; i < 32; i++) {
    //     mpz_setbit(num, i * 32 + 31);  // 每32位设置高位为1
    // }
}

typedef struct {
  mpz_t x0, x1, x2;
  mpz_t o0, o1;
  mpz_t w0, w1;
  mpz_t s0;
  mpz_t r;
} g_data_t;


g_data_t *g_generate_data(gmp_randstate_t state, uint32_t size, uint32_t count) {
  g_data_t *data=(g_data_t *)malloc(sizeof(g_data_t)*count);
  for(int index=0;index<count;index++) {
    mpz_init(data[index].x0);
    mpz_init(data[index].x1);
    mpz_init(data[index].x2);
    mpz_init(data[index].o0);
    mpz_init(data[index].o1);
    mpz_init(data[index].w0);
    mpz_init(data[index].w1);
    mpz_init(data[index].s0);
    mpz_init(data[index].r);
  }

  for(int index=0;index<count;index++) {
    mpz_urandomb(data[index].x0, state, size);
    mpz_urandomb(data[index].x1, state, size);
    mpz_urandomb(data[index].x2, state, size);
    mpz_urandomb(data[index].o0, state, size);
    mpz_urandomb(data[index].o1, state, size);
    mpz_urandomb(data[index].w0, state, 2*size);
    mpz_urandomb(data[index].w1, state, 2*size);
    mpz_urandomb(data[index].s0, state, 512);
    mpz_setbit(data[index].o0, 0);
    mpz_setbit(data[index].o1, 0);
  }
  return data;
}

void g_free_data(g_data_t *data, uint32_t count) {
  for(int index=0;index<count;index++) {
    mpz_clear(data[index].x0);
    mpz_clear(data[index].x1);
    mpz_clear(data[index].x2);
    mpz_clear(data[index].o0);
    mpz_clear(data[index].o1);
    mpz_clear(data[index].w0);
    mpz_clear(data[index].w1);
    mpz_clear(data[index].s0);
    mpz_clear(data[index].r);
  }
  free(data);
}
// 测试函数
void benchmark(size_t size, int bits) {
    // vector<mpz_t> a(size), b(size), e(size), res(size), r(size);
    g_data_t *data = g_generate_data(gmp_randstate, bits, size);
    // 设置线程数
    omp_set_num_threads(NUM_THEADS);

    // warm up
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_add(data[i].r, data[i].x0, data[i].x1);
    }
    // 加法
    auto start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_add(data[i].r, data[i].x0, data[i].x1);
    }
    auto end = omp_get_wtime();
    double ops = size / (end - start);
    printf("Addition bits %d: %.f op/s \n", bits, ops);

    // 减法
    start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_sub(data[i].r, data[i].x0, data[i].x1);
    }
    end = omp_get_wtime();
    ops = size / (end - start);
    printf("Subtraction bits %d: %.f op/s \n", bits, ops);
    // 乘法
    start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_mul(data[i].r, data[i].x0, data[i].x1);
    }
    end = omp_get_wtime();
    ops = size / (end - start);
    printf("Multiplication bits %d: %.f op/s \n", bits, ops);

    // 除法
    start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_tdiv_qr(data[i].o1, data[i].r, data[i].x0, data[i].x1);
    }
    end = omp_get_wtime();
    ops = size / (end - start);
    printf("Division bits %d: %.f op/s \n", bits, ops);

    // 模乘
    start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_mul(data[i].r, data[i].x0, data[i].x1);
        mpz_mod(data[i].r, data[i].r, data[i].o0);
    }
    end = omp_get_wtime();
    ops = size / (end - start);
    printf("Modular multiplication bits %d: %.f op/s \n", bits, ops);

    // 模幂
    int count = size;
    if (size >= 1024) size = 128 * 1024 / bits; 
    start = omp_get_wtime();
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        mpz_powm(data[i].r, data[i].x0, data[i].s0, data[i].o0);
    }
    end = omp_get_wtime();
    ops = size / (end - start);
    printf("Modular exponentiation bits %d: %.f op/s \n", bits, ops);
    size = count;

    // 释放内存
    g_free_data(data, size);
}

int main() {
    gmp_randinit_default(gmp_randstate);
    // gmp_randseed_ui(gmp_randstate, time(NULL));
    vector<size_t> bits_vec = {128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192};
    for (auto bits: bits_vec) {
        benchmark(400000, bits);
        cout << endl;
    }

    gmp_randclear(gmp_randstate);
    return 0;
}

