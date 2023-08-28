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
#include "layerality_instance.hpp"

namespace core {
	device::device()
	{
		cudaDeviceProp prop;
		assert_err(cudaGetDeviceProperties(&prop, 0));
		assert_err(cudaSetDevice(0));
		console("init device: " + std::string(prop.name));
		console("total const memory: " + std::to_string(prop.totalConstMem));
		tables::init();

	}

	device::~device() {
		cudaDeviceReset();
		console("reset device");
	}
}