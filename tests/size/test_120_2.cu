
#include "test.cuh"
std::string do_test3_120_2() { return do_test3<static_cast<std::uint32_t>((1 << 24) * 1.20 + 1 - 1e-10), 2>(); }
