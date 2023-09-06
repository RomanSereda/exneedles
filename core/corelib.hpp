#pragma once
#define LIBRARY "core"
#include "../deflib.inc"

namespace core
{
	class LIBRARY_API isystem {
	public:

	};
}

class LIBRARY_API corelib {
public:
	corelib();
	~corelib();

	const core::isystem& system() const;

private:
	core::isystem* m_system = nullptr;
};



