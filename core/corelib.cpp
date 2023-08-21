#include "corelib.hpp"
#include <cuda_runtime.h>
#include <string>
#include "memory.cuh"
#include "assert.hpp"
#include "instance.hpp"
#include "layerality.hpp"

namespace core {
	ptree test_ptree_cltr() {
		ptree root;
		innate::cluster_targeted cl;
		cl.width = 8;
		cl.height = 8;
		cl.target_layer = 3;
		cl.target_region = 9;

		innate::synapse_simple tr;

		return instance::terminality<UPTR_TEMPLATE_TR>::
			to_ptree((innate::cluster*)&cl, (innate::terminal*)&tr);
	}

	void test_cltr() {
		innate::layer layer {128, 128, 1};

		auto root = test_ptree_cltr();
		instance::host_terminality htr(root, layer);
		instance::device_terminality dtr(htr.to_ptree(), layer);

		memory::test_mempart_cltr(dtr.const_emplace_cl(), dtr.const_emplace_tr());
	}

	device::device()
	{
		cudaDeviceProp prop;
		assert_err(cudaGetDeviceProperties(&prop, 0));
		assert_err(cudaSetDevice(0));
		console("init device: " + std::string(prop.name));
		console("total const memory: " + std::to_string(prop.totalConstMem));
		tables::init();

		test_cltr();
	}

	device::~device() {
		cudaDeviceReset();
		console("reset device");
	}
}