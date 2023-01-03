
#include "lookup/test.cuh"
#include "correctness/test.cuh"

int main() {
  do_correctness_test();
  do_test2_all();
  RandomSetGenerator<(1 << 24)>::get()->free();
  return 0;
}