#pragma once
#include <map>
#include "corelib.hpp"
#include "boost.hpp"

namespace instance {
	class region_host;
	class region_device;
}

namespace core {
	class system : public isystem {
	public:
		system();
		~system();

	private:
		instance::region_host*   host   = nullptr;
		instance::region_device* device = nullptr;
	};
}
