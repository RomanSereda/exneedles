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

namespace dev_const_mem {
	struct deleter {
		void operator()(void* data) const noexcept;
	};
	std::unique_ptr<void, dev_const_mem::deleter> make_ptr(std::size_t size);

	struct offset {
		using ptr = std::shared_ptr<offset>;

		std::unique_ptr<void, dev_const_mem::deleter> hostmem;
		size_t szb;
		size_t value;

		__const__ void* p = nullptr;
	};

	__host__ offset::ptr __add_mempart(void* t, size_t szb);
	template<typename T> extern offset::ptr add_mempart(T* t) {
		return __add_mempart(t, sizeof(T));
	}
}

namespace helper {
	__device__ void print(int d);
	__device__ void print(float f);
}