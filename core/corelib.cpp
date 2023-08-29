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
	system::system()
	{
		cudaDeviceProp prop;
		assert_err(cudaGetDeviceProperties(&prop, 0));
		assert_err(cudaSetDevice(0));
		console("init device: " + std::string(prop.name));
		console("total const memory: " + std::to_string(prop.totalConstMem));
		tables::init();

	}

	system::~system() {
		cudaDeviceReset();
		console("reset device");
	}

	const lib_instance_host_type* system::host_region() const {
		return m_host_region.get();
	}
	
	const lib_instance_device_type* system::device_region() const {
		return m_device_region.get();
	}
}