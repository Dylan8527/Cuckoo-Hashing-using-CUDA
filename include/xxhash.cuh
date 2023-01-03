#ifndef __XXHASH_CUH__
#define __XXHASH_CUH__

// Reference website: https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h LINE 2159 
 /* #define instead of static const, to be used as initializers */
#define PRIME32_1  0x9E3779B1U  /*!< 0b10011110001101110111100110110001 */
#define PRIME32_2  0x85EBCA77U  /*!< 0b10000101111010111100101001110111 */
#define PRIME32_3  0xC2B2AE3DU  /*!< 0b11000010101100101010111000111101 */
#define PRIME32_4  0x27D4EB2FU  /*!< 0b00100111110101001110101100101111 */
#define PRIME32_5  0x165667B1U  /*!< 0b00010110010101100110011110110001 */

// a bit manipulation operator that is used to rotate the bits of a value to the left by a specified number of places
// For example: rotate_left(0b00000001, 1) = 0b00000010
__host__ __device__ __forceinline__ std::uint32_t rotate_left(std::uint32_t v,
                                                              std::uint32_t n) {
    return (v << n) | (v >> (32 - n));
}

// a function that is used to mix the bits of a value
// seed: one seed only for one hash function, i.e. we use seed to generate different hash functions
template <std::uint32_t seed>
__host__ __device__ __forceinline__ std::uint32_t xxhash(std::uint32_t v) {
    // Reference website: https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h LINE 2356 XXH32_endian_align
    std::uint32_t hash = seed + PRIME32_5;
    hash += 4u;

    // Reference website: https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h LINE 2267 XXH32_finalize
    /* Compact rerollled version; generally faster */
    //! slow version
        // hash +=  v * PRIME32_3;
        // hash  = rotate_left(hash, 17) * PRIME32_4;
    //! fast version
    auto bytes = reinterpret_cast<std::uint8_t *>(&v);
    for (std::uint32_t i = 0; i < 4; i++) {
        hash += bytes[i] * PRIME32_5;
        hash = rotate_left(hash, 11) * PRIME32_1;
    }

    // Reference website: https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h LINE 2239 XXH32_avalanche
    hash ^= hash >> 15;
    hash *= PRIME32_2;
    hash ^= hash >> 13;
    hash *= PRIME32_3;
    hash ^= hash >> 16;

    return hash;
}

#endif