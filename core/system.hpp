#pragma once
#include "corelib.hpp"
#include <memory>

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
		std::unique_ptr<instance::region_host>   host   {nullptr};
		std::unique_ptr<instance::region_device> device {nullptr};
	};
}
