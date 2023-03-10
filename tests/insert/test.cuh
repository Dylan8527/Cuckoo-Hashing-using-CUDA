//
// Created by xehoth on 2021/12/12.
//

#ifndef CS121_LAB2_TESTS_INSERT_TEST_CUH_
#define CS121_LAB2_TESTS_INSERT_TEST_CUH_
#include <cstdint>
#include <cuda_hash_table.cuh>
#include <rng.cuh>
#include <timer.cuh>
#include <cstdio>
#include <fstream>

template <std::uint32_t s, std::uint32_t N_H>
std::string do_test1() {
  fprintf(stderr, "test1 (s = %u, t = %u):\n", s, N_H);
  constexpr std::uint32_t C = 1 << 25;
  constexpr std::uint32_t S = 1 << s;
  HashTable<C, 4 * s, N_H> table;
  DeviceArray<std::uint32_t, S> d_set;
  fprintf(stderr, "  generate random set ... ");
  RandomSetGenerator<(1 << 24)>::get()->generate_random_set<S>(d_set);
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
  printf("%-4u%-4u", s, N_H);
  timer.report(S);
  fflush(stdout);
  fprintf(stderr, "done\n\n");
  return timer.to_string(S);
}

void do_test1_all();
#endif  // CS121_LAB2_TESTS_INSERT_TEST_CUH_
