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

namespace memory {
	struct deleter {
		void operator()(void* data) const noexcept;
	};
	using void_uptr = std::unique_ptr<void, memory::deleter>;
	void_uptr make_void_uptr(std::size_t size);

	struct const_empl {
		using ptr = std::shared_ptr<const_empl>;

		void_uptr duplicate;
		size_t szb;
		size_t offset;

		__const__ void* const_ptr = nullptr;
	};

	__host__ const_empl::ptr __add_mempart(void* t, size_t szb);
	template<typename T> extern const_empl::ptr add_mempart(T* t) {
		return __add_mempart(t, sizeof(T));
	}

	void test_mempart_cltr(const const_empl::ptr& ptr_cl, const const_empl::ptr& ptr_tr);
}

namespace helper {
	__device__ void print(int d);
	__device__ void print(float f);
}