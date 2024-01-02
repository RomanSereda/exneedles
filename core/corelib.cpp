#include "corelib.hpp"
#include <cuda_runtime.h>
#include <string>
#include "memory.cuh"
#include "assert.hpp"

#include "system.hpp"

corelib::corelib()
{
	cudaDeviceProp prop;
	assert_err(cudaGetDeviceProperties(&prop, 0));
	assert_err(cudaSetDevice(0));
	console("init device: " + std::string(prop.name));
	console("total const memory: " + std::to_string(prop.totalConstMem));
	tables::init();

	m_system = new core::system();
}

corelib::~corelib() {
	if (m_system)
		delete m_system;

	cudaDeviceReset();
	console("reset device");
}

core::isystem& corelib::system() {
	return *m_system;
}