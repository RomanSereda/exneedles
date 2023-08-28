#include "memory.cuh"
#include <time.h>
#include <stdio.h>

#include <list>
#include <string> 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "assert.hpp"

#include "terminality.hpp" // for test
#include "cellularity.hpp" // for test

namespace tables {
	void init() {
		init_table_nbits_values();
		init_table_stdp_values();
		init_table_rand_values();
	}

	const uint8_t const8 = 8;
	rgstr8_t tact_inc8(int& inc) {
		rgstr8_t h_inc8 = 0;

		if (inc == const8 * const8 * const8 * const8 * const8 * const8 * const8 * const8) {
			inc = 0;
		}

		if (inc == 0) {
			inc++;
			return h_inc8;
		}

		if (inc % const8 == 0) {
			h_inc8 = h_inc8 | 0b00000001;

			if (inc % (const8 * const8) == 0) {
				h_inc8 = h_inc8 | 0b00000010;

				if (inc % (const8 * const8 * const8) == 0) {
					h_inc8 = h_inc8 | 0b00000100;

					if (inc % (const8 * const8 * const8 * const8) == 0) {
						h_inc8 = h_inc8 | 0b00001000;

						if (inc % (const8 * const8 * const8 * const8 * const8) == 0) {
							h_inc8 = h_inc8 | 0b00010000;

							if (inc % (const8 * const8 * const8 * const8 * const8 * const8) == 0) {
								h_inc8 = h_inc8 | 0b00100000;

								if (inc % (const8 * const8 * const8 * const8 * const8 * const8 * const8) == 0) {
									h_inc8 = h_inc8 | 0b01000000;

									if (inc % (const8 * const8 * const8 * const8 * const8 * const8 * const8 * const8) == 0) {
										h_inc8 = h_inc8 | 0b10000000;
									}
								}
							}
						}
					}
				}
			}
		}

		inc++;

		return h_inc8;
	}

	const int sz_stdp_values = 128 * 128;
	__constant__ int8_t stdp_values_table[sz_stdp_values];
	__device__ float stdp(const uint8_t cell_spikes,
		const uint8_t synapse_spikes)
	{
		if (cell_spikes < 128)
			if (synapse_spikes < 128) {
				int stdp_index = (int)cell_spikes
					+ ((int)128) * synapse_spikes;
				return ((float)stdp_values_table[stdp_index]) / 14;
			}
		return 0;
	}
	float comp_stdp(const uint8_t cell_spikes,
		const uint8_t synapse_spikes)
	{
		float sum_dt = 0;

		for (int i = 0; i < 8; ++i) {
			if (cell_spikes & (1 << i)) {
				for (int j = 0; j < 8; ++j) {
					if (synapse_spikes & (1 << j)) {
						float d = (float)j - i;
						if (d == 0) d = 0.5;

						d = 1 / d;

						sum_dt += (7 - i) * d;
					}
				}
			}
		}

		return sum_dt / 7;
	};
	void init_table_stdp_values() {
		if (int8_t* buf = (int8_t*)malloc(sz_stdp_values)) {
			for (int i = 0; i < sz_stdp_values; i++) {
				uint8_t x = i % 128;
				uint8_t y = i / 128;

				float val = comp_stdp(x, y);

				buf[x + 128 * y] = (int8_t)val;
			}
			assert_err(cudaMemcpyToSymbol(stdp_values_table, buf,
				sz_stdp_values * sizeof(int8_t)));

			free(buf);

			console("getted const mem: " + std::to_string(sz_stdp_values));
		}
		else logexit();
	}

	const int sz_nbits_values = 256;
	__constant__ uint8_t nbits_values_table[sz_nbits_values];
	__device__ uint8_t nbits(const uint8_t n) {
		return nbits_values_table[n];
	}
	uint8_t comp_nbits(uint8_t n) {
		uint8_t res = 0;
		while (n) {
			res++;
			n &= n - 1;
		}
		return res;
	}
	void init_table_nbits_values() {
		if (int8_t* buf = (int8_t*)malloc(sz_nbits_values)) {
			for (int i = 0; i < sz_nbits_values; i++)
				buf[i] = comp_nbits(i);

			assert_err(cudaMemcpyToSymbol(nbits_values_table, buf,
				sz_nbits_values * sizeof(int8_t)));
			free(buf);

			console("getted const mem: " + std::to_string(sz_nbits_values));
		}
		else logexit();
	}

	const uint sz_rand_coseed = 1024;
	__constant__ uint rand_coseed[sz_rand_coseed];
	void init_table_rand_values() {
		if (uint* h_rand_coseed = (uint*)malloc(sz_rand_coseed * sizeof(uint))) {
			srand((uint)time(NULL));

			for (size_t i = 0; i < sz_rand_coseed; i++)
				h_rand_coseed[i] = rand();

			assert_err(cudaMemcpyToSymbol(rand_coseed, h_rand_coseed,
				sz_rand_coseed * sizeof(uint), 0, cudaMemcpyHostToDevice));

			free(h_rand_coseed);

			console("getted const mem: " + std::to_string(sz_rand_coseed));
		}
		else logexit();
	}
	__device__ unsigned int curand() {
		unsigned int r = blockIdx.x * blockDim.x + threadIdx.x + clock();
		return rand_coseed[r % sz_rand_coseed];
	}
	__device__ unsigned int static_curand() {
		unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
		return rand_coseed[r % sz_rand_coseed];
	}
}

namespace memory {
	void memory::deleter::operator()(void* data) const noexcept {
		std::free(data);
	}

	void_uptr memory::make_void_uptr(std::size_t size) {
		return void_uptr(std::malloc(size));
	}

	const int sz_const_pool = 16384;
	__constant__ uint8_t const_pool_table[sz_const_pool];
	std::list<const_empl::ptr> parts;

	const_empl::ptr __add_mempart(void* t, size_t szb) {
		const_empl::ptr mempart = nullptr;
		if (auto ptr_host_mem = memory::make_void_uptr(szb)) {
			memcpy(ptr_host_mem.get(), t, szb);

			size_t value = 0;
			if (parts.empty()) value = 0;
			else value = parts.back()->offset + parts.back()->szb;

			mempart = std::make_shared<const_empl>(
				memory::const_empl{ std::move(ptr_host_mem), szb, value });

			parts.push_back(mempart);
		}
		else logexit();

		if (!mempart)
			logexit();

		return mempart;
	}

	__host__ void remove_mempart(const const_empl::ptr& ptr)
	{
		parts.remove(ptr);
	}

	__host__ void setup_const_memoryparts()
	{
		if (uint8_t* temp_table = (uint8_t*)malloc(sz_const_pool)) {
			for (const auto& part : parts) {
				if (!memcpy(&temp_table[part->offset], part->duplicate.get(), part->szb))
					logexit();
			}
			assert_err(cudaMemcpyToSymbol(const_pool_table, temp_table, sz_const_pool));
			free(temp_table);

			void* const_mem_address = nullptr;
			assert_err(cudaGetSymbolAddress((void**)&const_mem_address, const_pool_table));

			if (!const_mem_address)
				logexit();

			for (auto& part : parts)
				part->const_ptr = (void*)((size_t)const_mem_address + part->offset);
		}
		else logexit();
	}

	__global__ void test_mempart_kernel_cltr(const innate::cluster_targeted* cl,
		                                     const innate::synapse_simple* tr) {
		printf("cluster height: %d\n", cl->height);
		printf("cluster width: %d\n", cl->width);

		printf("cluster target_layer: %d\n", cl->target_layer_index);
		printf("cluster target_spillover: %d\n", cl->target_spillover_index);

		printf("synapse sign: %d\n", tr->sign);
		printf("synapse type: %d\n", tr->type);
	}
	void test_mempart_cltr(const memory::const_empl::ptr& ptr_cl,
		                                       const const_empl::ptr& ptr_tr) {
		test_mempart_kernel_cltr <<<1, 1 >>> ((const innate::cluster_targeted*)ptr_cl->const_ptr,
			                                  (const innate::synapse_simple*)ptr_tr->const_ptr);
	}
	__global__ void test_mempart_kernel_cell(const innate::cell_exre* c) {
		printf("cell tacts_excitation: %d\n", c->tacts_excitation);
		printf("cell tacts_relaxation: %d\n", c->tacts_relaxation);
	}
	void test_mempart_cell(const memory::const_empl::ptr& ptr_c) {
		test_mempart_kernel_cell << <1, 1 >> > ((const innate::cell_exre*)ptr_c->const_ptr);
	}
}

namespace helper {
	__device__  void print(int d) {
		printf("blk %d th %d, %d\n", blockIdx.x, threadIdx.x, d);
	}
	__device__ void print(float f) {
		printf("blk %d th %d, %f\n", blockIdx.x, threadIdx.x, f);
	}
}