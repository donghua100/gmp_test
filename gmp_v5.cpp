#include <gmp.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cassert>

#define NUM_THREADS 32
static gmp_randstate_t gmp_randstate;

// 数据结构
typedef struct {
  mpz_t x0, x1, res, rem, exp, m, base;
} g_data_t;

typedef enum {
        add,
        sub,
        mul,
        div_pr,
        mulmod,
        expmod,
} Op;


// 生成随机数据
g_data_t* g_generate_data(gmp_randstate_t state, uint32_t size, uint32_t count) {
  g_data_t* data = (g_data_t*)malloc(sizeof(g_data_t) * count);
  for (uint32_t i = 0; i < count; i++) {
    mpz_init(data[i].x0);
    mpz_init(data[i].x1);
    mpz_init(data[i].res);

    mpz_init(data[i].rem);
    mpz_init(data[i].exp);
        mpz_init(data[i].m);
        mpz_init(data[i].base);

    mpz_urandomb(data[i].x0, state, size);
    mpz_urandomb(data[i].x1, state, size);
    mpz_urandomb(data[i].m, state, size);
    mpz_urandomb(data[i].exp, state, 512);
    mpz_urandomb(data[i].base, state, size);
    mpz_setbit(data[i].m, 0);  // 确保模数不为0
  }
  return data;
}

void g_free_data(g_data_t* data, uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    mpz_clear(data[i].x0);
    mpz_clear(data[i].x1);
    mpz_clear(data[i].res);
    mpz_clear(data[i].rem);
    mpz_clear(data[i].exp);
        mpz_clear(data[i].base);
        mpz_clear(data[i].m);
  }
  free(data);
}

double test_add(int count, int bits, g_data_t *data) {
  int LOOPS = 1;
  #pragma omp parallel for
  for (uint32_t i = 0; i < count; i++) {
    mpz_t priv_t;
    mpz_init(priv_t);
    for (uint64_t loop = 0; loop < LOOPS; loop++) {
      mpz_add(priv_t, data[i].x0, data[i].x1);
    }
    mpz_set(data[i].res, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS * count;

}
double test_sub(int count, int bits, g_data_t *data) {
  int LOOPS = 1;
  #pragma omp parallel for
  for (uint32_t i = 0; i < count; i++) {
    mpz_t priv_t;
    mpz_init(priv_t);
    for (uint64_t loop = 0; loop < LOOPS; loop++) {
      mpz_sub(priv_t, data[i].x0, data[i].x1);
    }
    mpz_set(data[i].res, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS * count;

}
double test_mul(int count, int bits, g_data_t *data) {
  int LOOPS = 1;
  #pragma omp parallel for
  for (uint32_t i = 0; i < count; i++) {
    mpz_t priv_t;
    mpz_init(priv_t);
    for (uint64_t loop = 0; loop < LOOPS; loop++) {
      mpz_mul(priv_t, data[i].x0, data[i].x1);
    }
    mpz_set(data[i].res, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS * count;

}
double test_div(int count, int bits, g_data_t *data) {
  int LOOPS = 1;
  #pragma omp parallel for
  for (uint32_t i = 0; i < count; i++) {
    mpz_t priv_t, priv_t1;
    mpz_init(priv_t);
        mpz_init(priv_t1);
    for (uint64_t loop = 0; loop < LOOPS; loop++) {
      mpz_tdiv_qr(priv_t,  priv_t1, data[i].x0, data[i].x1);
    }
    mpz_set(data[i].res, priv_t);
    mpz_set(data[i].rem, priv_t1);
    mpz_clear(priv_t);
    mpz_clear(priv_t1);
  }
  return LOOPS * count;

}

double test_mulmod(int count, int bits, g_data_t *data) {
  int LOOPS = 1;
  #pragma omp parallel for
  for (uint32_t i = 0; i < count; i++) {
    mpz_t priv_t;
    mpz_init(priv_t);
    for (uint64_t loop = 0; loop < LOOPS; loop++) {
      mpz_mul(priv_t, data[i].x0, data[i].x1);
          mpz_mod(priv_t, priv_t, data[i].m);
    }
    mpz_set(data[i].res, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS * count;

}
double test_expmod(int count, int bits, g_data_t *data) {
  int LOOPS = 1;
  #pragma omp parallel for
  for (uint32_t i = 0; i < count; i++) {
    mpz_t priv_t;
    mpz_init(priv_t);
    for (uint64_t loop = 0; loop < LOOPS; loop++) {
      mpz_powm(priv_t, data[i].base, data[i].exp, data[i].m);
    }
    mpz_set(data[i].res, priv_t);
    mpz_clear(priv_t);
  }
  return LOOPS * count;

}
const char *s[6] = {"Addition", "Subtraction", "Multiplication", "Division", "Modular Multiplication", "Modular Exponentiation"};

// 通用模板函数
double benchmark_op(uint32_t bits, g_data_t* data, uint32_t count, Op op) {
        double cnt = 0;
        if (op == add) {
                cnt = test_add(count, bits, data);
        }
        else if (op == sub) {
                cnt = test_sub(count, bits, data);
        }
        else if (op == mul) {
                cnt = test_mul(count, bits, data);
        }
        else if (op == div_pr) {
                cnt = test_div(count, bits, data);
        }
        else if (op == mulmod) {
                cnt = test_mulmod(count, bits, data);
        }
        else if (op == expmod) {
                cnt = test_expmod(count, bits, data);
        }
        else assert(0);
        return cnt;
}

// 运行测试
void run_benchmark(uint32_t bits, uint32_t count) {
  omp_set_num_threads(NUM_THREADS);

  double start, end, ops;
  uint32_t c = count;
  for (int op = 0; op < 6; op++) {
        if (op == 0 || op == 1) count = 1000 * c;
        if (op == 2 || op == 3 || op == 4) count = 10 * c;
        if (op == 5) count = 1*c;
        g_data_t* data = g_generate_data(gmp_randstate, bits, count);
        int wm = 5;
        if (op == 5) wm = 1;
        for (int i = 0; i < wm; i++) {
            benchmark_op(bits, data, count, (Op)op);
        }
        int repate = 100;
        if (op == 5) repate = 2;
        start = omp_get_wtime();
        double cnt = 0;
        for (int i = 0; i <  repate; i++) {
          cnt = benchmark_op(bits, data, count, (Op)op);
        }
          end = omp_get_wtime();
          ops = repate * cnt / (end - start);
          printf("%s bits %u: %.f op/s\n",s[op], bits, ops);
    g_free_data(data, count);

  }

}

int main() {
  gmp_randinit_default(gmp_randstate);
  // gmp_randseed_ui(gmp_randstate, time(NULL));

  std::vector<uint32_t> bits_vec = {128,256,512,1024,2048,3072,4096,5120,6144,7168,8192};
  uint32_t count = 1000;

  for (auto bits : bits_vec) {
    run_benchmark(bits, count);
    std::cout << std::endl;
  }

  gmp_randclear(gmp_randstate);
  return 0;
}



