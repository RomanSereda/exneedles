#include "corelib.hpp"
#include <cuda_runtime.h>
#include <string>
#include "memory.cuh"
#include "assert.hpp"

#include "terminality.hpp"
#include "cellularity.hpp"
#include "layerality.hpp"

#include "terminality_instance.hpp"
#include "cellularity_instance.hpp"

namespace core {
	ptree test_ptree_cltr() {
		innate::cluster_targeted cl;
		cl.width = 8;
		cl.height = 8;
		cl.target_layer = 3;
		cl.target_region = 9;

		innate::synapse_simple tr;

		return instance::terminality<UPTR_TEMPLATE_TR>::
			to_ptree((innate::cluster*)&cl, (innate::terminal*)&tr);
	}

	ptree test_ptree_cell() {
		innate::cell_exre c;
		c.tacts_excitation = 2;
		c.tacts_relaxation = 3;

		return instance::cellularity_cpu_type::to_ptree((innate::cell*)&c);
	}

	void test() {
		innate::layer layer {128, 128, 1};

		auto root = test_ptree_cltr();
		instance::terminality_host htr(root, layer);
		instance::terminality_device dtr(htr.to_ptree(), layer);
		memory::test_mempart_cltr(dtr.const_emplace_cl(), dtr.const_emplace_tr());

		instance::cellularity_host hcr(test_ptree_cell(), layer);
		auto cell_host_ptree = hcr.to_ptree();
		std::vector<instance::terminality_host*> ths;
		ths.push_back(&htr);
		ths.push_back(&htr);
		boost::add_array(cell_host_ptree, "terminalitys", ths);
		
		instance::cellularity_host hcr2(cell_host_ptree, layer);
		instance::cellularity_device dcr(hcr2.to_ptree(), layer);
		memory::test_mempart_cell(dcr.const_emplace_cell());

		console(boost::to_string(dcr.to_ptree()));
		auto d = dcr.innate();

		int t = 0;

	}

	device::device()
	{
		cudaDeviceProp prop;
		assert_err(cudaGetDeviceProperties(&prop, 0));
		assert_err(cudaSetDevice(0));
		console("init device: " + std::string(prop.name));
		console("total const memory: " + std::to_string(prop.totalConstMem));
		tables::init();

		test();
	}

	device::~device() {
		cudaDeviceReset();
		console("reset device");
	}
}