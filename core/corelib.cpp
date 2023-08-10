#include "corelib.hpp"
#include <cuda_runtime.h>
#include "private/assert.hpp"

#include "types.hpp"
#include "terminality.hpp"

namespace core
{
	device::device()
	{
		cudaDeviceProp prop;
		assert_err(cudaGetDeviceProperties(&prop, 0));
		assert_err(cudaSetDevice(0));
		console("inited device " + std::string(prop.name));

	}

	device::~device() {
		cudaDeviceReset();
		console("reset device");

		//instance::cluster cluster;
	}
}