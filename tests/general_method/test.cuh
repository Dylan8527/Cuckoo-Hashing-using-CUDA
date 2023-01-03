#ifndef _GENERAL_TEST_CUH_
#define _GENERAL_TEST_CUH_
#include <cstdint>
#include <cuda_hash_table.cuh>
#include <rng.cuh>
#include <timer.cuh>
#include <cstdio>
#include <fstream>

template <uint C, uint t, uint S, uint bound>
std::string do_general_test() {
  fprintf(stderr, "do general test for C = %u, t = %u, S = %u, bound = %u", C, t, S, bound);

  // Generate hash table
  HashTable<C, bound, t> table;
  // Generate cuda array on gpu
  DeviceArray<uint, S> d_set;
  fprintf(stderr, "  generate random set ... ");
  // Generate S keys to insert
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
#endif  // _GENERAL_TEST_CUH_
