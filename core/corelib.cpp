#include "corelib.hpp"
#include <cuda_runtime.h>
#include <string>
#include "memory.cuh"
#include "assert.hpp"
#include "instance.hpp"
#include "layerality.hpp"

namespace core
{
	device::device()
	{
		cudaDeviceProp prop;
		assert_err(cudaGetDeviceProperties(&prop, 0));
		assert_err(cudaSetDevice(0));
		console("init device: " + std::string(prop.name));
		console("total const memory: " + std::to_string(prop.totalConstMem));
		tables::init();

		instance::host_terminality htr;

		auto root = htr.to_ptree();
		innate::layer layer {128, 128, 1};
		instance::device_terminality dtr(root, layer);
	}

	device::~device() {
		cudaDeviceReset();
		console("reset device");
	}
}