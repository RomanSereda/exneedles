#include "corelib.hpp"
#include <cuda_runtime.h>
#include "tables.cuh"
#include "assert.hpp"

namespace core
{
	device::device()
	{
		cudaDeviceProp prop;
		assert_err(cudaGetDeviceProperties(&prop, 0));
		assert_err(cudaSetDevice(0));
		console("init device " + std::string(prop.name));
		tables::init();
	}

	device::~device() {
		cudaDeviceReset();
		console("reset device");
	}
}