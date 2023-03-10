//
// Created by xehoth on 2021/12/13.
//

#ifndef CS121_LAB2_TESTS_SIZE_TEST_CUH_
#define CS121_LAB2_TESTS_SIZE_TEST_CUH_
#include <cuda_hash_table.cuh>
#include <rng.cuh>
#include <fstream>

template <std::uint32_t C, std::uint32_t N_H>
std::string do_test3() {
  fprintf(stderr, "test3 (ratio = %.2f, t = %u):\n",
          static_cast<double>(C) / (1 << 24), N_H);
  HashTable<C, 4 * 24, N_H> table;
  constexpr std::uint32_t S = 1 << 24;
  DeviceArray<std::uint32_t, S> d_set;
  fprintf(stderr, "  generate random set ... ");
  RandomSetGenerator<S>::get()->generate_random_set<S>(d_set);
  fprintf(stderr, "done\n");
  fprintf(stderr, "  begin testing ... \n");
  Timer timer;
  for (int i = 0; i < 5; ++i) {
    fprintf(stderr, "    round %d begin ... ", i);
    timer.start();
    table.insert(d_set);
    timer.end();
    fprintf(stderr, "done\n");
    table.clear();
  }
  fprintf(stderr, "  done\n");
  d_set.free();
  table.free();
  printf("%-6.2f%-4u", static_cast<double>(C) / (1 << 24), N_H);
  timer.report(S);
  fprintf(stderr, "done\n\n");
  return timer.to_string(S);
}

void do_test3_all();
#endif  // CS121_LAB2_TESTS_SIZE_TEST_CUH_
