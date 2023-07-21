#pragma once
#ifdef CORE_EXPORTS                               
#define LIB_EXPORTS  
#else
#define LIBRARY "core"
#endif
#include "../deflib.inc"

namespace core
{
	class LIBRARY_API device {
	public:
		device();
		~device();
	};
}

