#pragma once
#include <cuda_runtime.h>
#include "types.hpp"

namespace tables {
	void init();
	rgstr8_t tact_inc8(int& inc);

	void init_table_stdp_values();
	__device__ float stdp(const uint8_t cell_spikes,
		const uint8_t synapse_spikes);

	void init_table_nbits_values();
	__device__ uint8_t nbits(const uint8_t n);

	void init_table_rand_values();
	__device__ uint curand();
	__device__ uint static_curand();

	void init_table_const_pool();
	void* get_new_pool_part(void* t, size_t szb);
	template<typename T> extern T* get_new_pool_part(T* t) {
		return static_cast<T*>(get_new_pool_part(t, sizeof(T)));
	}
}

namespace helper {
	__device__ void print(int d);
	__device__ void print(float f);
}