#include "corelib.hpp"
#include <cuda_runtime.h>

namespace core
{
	device::device()
	{
		cudaDeviceProp prop;
		cuassert(cudaGetDeviceProperties(&prop, 0));
		cuassert(cudaSetDevice(0));
		console("init device " + string(prop.name));

	}

	device::~device() {
		cudaDeviceReset();
		console(string("reset device"));
	}
}