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
}

namespace helper {
	__device__ void print(int d);
	__device__ void print(float f);
}