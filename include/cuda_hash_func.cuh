#ifndef __CUDA_HASH_FUNC__
#define __CUDA_HASH_FUNC__

#include <cstdint>
#include <xxhash.cuh>

#define SEED_PRIME32_1 7979717u
#define SEED_PRIME32_2 998244353u

// static uint seeds[] = {2, SEED_PRIME32_1+SEED_PRIME32_2, SEED_PRIME32_1*5+SEED_PRIME32_2};
// return: hash value for key k
// k: key
// hash_func: determined by the hash_func_id (influence the seed of the hash function)
template <std::uint32_t hash_func_id>
struct TemplateHash {
    static __device__ __host__ __forceinline__ std::uint32_t hash(std::uint32_t k) {
        if constexpr (hash_func_id == 0)
            return xxhash<2>(k);
        else if constexpr (hash_func_id == 1)
            return xxhash<SEED_PRIME32_1+SEED_PRIME32_2>(k);
        else {
            return xxhash<SEED_PRIME32_1*(hash_func_id+3)+SEED_PRIME32_2>(k);
        }
    }
};

#endif