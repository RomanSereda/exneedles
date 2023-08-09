#include "corelib.hpp"
#include <cuda_runtime.h>
#include "private/assert.hpp"
#include "types.hpp"

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
	}
}

namespace instance {
	int cluster::terminal_bytes_size() const {
		int size = -1;
		data::terminal::foreach(std::get<__const__ innate::terminal*>(innate), [&size](auto* p) {
			size = sizeof(*p);
			return true;
			});
		return size;
	}
}