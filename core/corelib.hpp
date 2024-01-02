#pragma once
#define LIBRARY "core"
#include "../deflib.inc"

namespace instance {
	class iregion;
};

namespace core {
	class LIBRARY_API isystem {
	public:
		virtual instance::iregion& region() = 0;

	};
};

class LIBRARY_API corelib {
public:
	corelib();
	~corelib();

	core::isystem& system();

private:
	core::isystem* m_system = nullptr;
};



