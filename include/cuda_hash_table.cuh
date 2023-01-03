#ifndef __CUDA_HASH_TABLE_CUH__
#define __CUDA_HASH_TABLE_CUH__

#include <cuda_array.cuh>
#include <cuda_hash_func.cuh>
#include <cuda_utils.cuh>
#include <timer.cuh>

// hash table contains:
// 1. C             : capacity for each slot (cuda array)
// 2. bound         : searching bound
// 3. t             : number of hash functions (slots)
// 4. value         : init constant value for slots
//! THESE PARAMETERS CAN NOT BE CHANGED
#define hash_table_head template<uint C,                         \
                                 uint bound,                     \
                                 uint t,                         \
                                 uint value>
// hash kernel contains:
// 1. C             : capacity
// 2. t             : number of hash functions (cuda arrays)
// 3. S             : size of cuda array
// 4. bound         : searching bound
//! ABOVE PARAMETERS CAN NOT BE CHANGED
// 5. hash_func_id  : which hash function to use
//! ONLY THIS PARAMETER CAN BE CHANGED i.e. hash_func_id determines which hash function to use
#define hash_kernel_head template<uint C,                         \
                                  uint t,                         \
                                  uint S,                         \
                                  uint bound,                     \
                                  uint hash_func_i>

template<uint C, uint bound, uint t=2, uint value=-1u>
class HashTable;

hash_kernel_head
__global__ void insert_kernel(HashTable<C, bound, t> table,
                              DeviceArray<std::uint32_t, S> keys);

hash_kernel_head
__global__ void lookup_kernel(HashTable<C, bound, t> table,
                              DeviceArray<std::uint32_t, S> keys,
                              DeviceArray<std::uint32_t, S> results);

hash_table_head
class HashTable {
public:
    DeviceArray<uint, C> slots[t];
    DeviceArray<uint, 1> collisions; // 

    __host__ HashTable() { clear(); }

    void clear() {
        for (auto &slot: slots) slot.template init<value>(); // init slots (cuda arrays)
        uint h_collisions(0);
        cudaMemcpy(collisions.data, &h_collisions, sizeof(uint), cudaMemcpyHostToDevice);
    }

    void free() {
        for (auto &slot: slots) slot.free();
        collisions.free();
    }

    // insert S keys into hash table using h_i
    template <uint S, uint hash_func_i = 0>
    __forceinline__ void insert(DeviceArray<uint, S> keys) {
        // try to insert keys into hash table
        insert_kernel<C, t, S, bound, hash_func_i%t>
            <<<ceil(S / static_cast<double>(BLOCKSIZE)), BLOCKSIZE>>>
        (*this, keys);
        // check if there is h_collisions
        uint h_collisions(0);
        cudaMemcpy(&h_collisions, this->collisions.data, sizeof(uint), cudaMemcpyDeviceToHost);
        // if no h_collisions, return
        if (h_collisions == 0) {
            return;
        }
        // if h_collisions, rehash, we need use another slot to insert keys
        // i.e. change to next hash function
        fprintf(stderr, "[rehash %d]", hash_func_i +1);
        clear();
        if constexpr (hash_func_i < 32) {
            insert<S, hash_func_i+1>(keys); //! recursive call, warning: may cause stack overflow
        } else {
            fprintf(stderr, "[rehash too many times, failed]");
            return;
        }
    }

    // Using h_i hash function to
    // 1. insert S keys in hash table
    // 2. lookup LS keys in hash table 
    // 3. record time using Timer
    template <uint S, uint LS, uint hash_func_i = 0>
    __forceinline__ void insert_and_lookup(
        DeviceArray<uint, S>  keys,                     // keys to insert
        DeviceArray<uint, LS> lookup_keys,              // keys to lookup
        DeviceArray<uint, LS> lookup_results,           // lookup results
        Timer &timer                                    // record time
    ) {
        //1. insert S keys
        insert_kernel<C, t, S, bound, hash_func_i%t>
            <<< ceil(S / static_cast<double>(BLOCKSIZE)), BLOCKSIZE>>>
        (*this, keys);
        uint h_collisions(0);
        cudaMemcpy(&h_collisions, this->collisions.data, sizeof(uint), cudaMemcpyDeviceToHost);
        // insert successfully and lookup LS keys
        if (h_collisions == 0) {
            //2. lookup LS keys
            timer.start();
            lookup_kernel<C, t, LS, bound, hash_func_i%t>
                <<< ceil(S / static_cast<double>(BLOCKSIZE)), BLOCKSIZE>>>
            (*this, lookup_keys, lookup_results);
            timer.end();
            return;
        }
        fprintf(stderr, "[rehash %d]", hash_func_i +1);
        clear();
        if constexpr (hash_func_i < 32) {
            insert_and_lookup<S, LS, hash_func_i+1>(keys, lookup_keys, lookup_results, timer);
        } else {
            fprintf(stderr, "[rehash too many times, failed]");
            return;
        }
    }

    [[nodiscard]] constexpr __device__ __host__ __forceinline__ uint 
    emptyKey() const {
        return value;
    }

    void print() {
        HostArray<uint, C> h_slots[t];
        fprintf(stderr, "Print the hash table...");
        // copy slots from device to host
        for (uint i = 0; i < t; ++i) h_slots[i] = slots[i];
        // print slots
        for (uint i = 0; i < t; ++i) 
            // cuda array for ith hash function~
            for (uint j = 0; j < C; ++j) 
                if (h_slots[i](j) != value) 
                    printf("%u: (%u, %u)\n", h_slots[i](j), i, j);
        // free slots
        for (auto &slot : h_slots) slot.free();
    }
};

// insert to cur position in ith slot
// 1. search_depth : the depth of search, start from 'bound' and decrease to 0
#define hash_func_head template<uint  search_depth>

hash_kernel_head
struct TemplateInsert {
    hash_func_head
    static __device__ __forceinline__ void insert(HashTable<C, bound, t> table, uint key) {
        if (table.collisions(0) != 0) return;
        // evict key at the cur'th slots, with position at [h_i(key) % C]
        uint last = atomicExch(
            &(table.slots[hash_func_i](TemplateHash<hash_func_i>::hash(key) % C)), key
        );
        // if last is empty or last is key(no need to insert key), no collision
        if (last == table.emptyKey() || last == key) {
            // no collision or already inserted
            return;
        }
        // if last is not empty, insert last to next slot
        else{
            if constexpr (search_depth > 0) {
                TemplateInsert<C, t, S, bound, (hash_func_i + 1)%t>::insert<search_depth-1>(table, last);
            }
            else {
                //! reach search bound, collision happened!
                // Here, we just modify the first element of collisions array,
                atomicAdd(&table.collisions(0), 1);
                // return and do not continue to search
            }
        }
    }
};


hash_kernel_head
__global__ void insert_kernel(HashTable<C, bound, t> table,
                              DeviceArray<uint, S> keys) {
    cuda_foreach_uint(x, 0, S) {
        TemplateInsert<C, t, S, bound, 0>::insert<bound>(table, keys(x));
    }
}

hash_kernel_head
struct TemplateLookup {
    static __device__ __forceinline__ bool lookup(HashTable<C, bound, t> table,
                                                  uint key) {
        // if key is not in the last slot, return false
        if constexpr (hash_func_i >= t) {
            return false;
        }
        else {
            if (table.slots[hash_func_i](TemplateHash<hash_func_i>::hash(key) % C) == key) { // if key is in the i'th slot, return
                return true;
            }
            // if key is not in the cur'th slot, lookup next slot
            return TemplateLookup<C, t, S, bound, hash_func_i+1>::lookup(table, key);
        }
    }
};


hash_kernel_head
__global__ void lookup_kernel(HashTable<C, bound, t> table,
                              DeviceArray<uint, S> keys,
                              DeviceArray<uint, S> results) {
    cuda_foreach_uint(x, 0, S) {
        results(x) = 
            TemplateLookup<C, t, S, bound, 0>::lookup(table, keys(x));
    }
}
#endif // __CUDA_HASH_TABLE_CUH__