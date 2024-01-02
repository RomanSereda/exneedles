#pragma once
#include <map>
#include "corelib.hpp"
#include "boost.hpp"

namespace instance {
	class iregion;
	class region_host;
	class region_device;
}

namespace core {
	class system : public isystem {
	public:
		system();
		~system();

		instance::iregion& region() override;

	private:
		instance::region_host*   host   = nullptr;
		instance::region_device* device = nullptr;
	};
}
