#include "corelib.hpp"
#include <cuda_runtime.h>
#include <string>
#include "tables.cuh"
#include "assert.hpp"
#include "terminality.hpp"

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
	}

	device::~device() {
		cudaDeviceReset();
		console("reset device");
	}
}