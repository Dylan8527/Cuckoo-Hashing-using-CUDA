
#include "test.cuh"
std::string do_test3_110_2() { return do_test3<static_cast<std::uint32_t>((1 << 24) * 1.10 + 1 - 1e-10), 2>(); }
