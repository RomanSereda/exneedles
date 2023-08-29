#pragma once
#ifdef CORE_EXPORTS                               
#define LIB_EXPORTS  
#else
#define LIBRARY "core"
#endif

#include "../deflib.inc"
#include "layerality.hpp"

#ifdef _CORE
	#include "layerality_instance.hpp"
	#define lib_instance_host_type   instance::region_host 
	#define lib_instance_device_type instance::region_device 
	#define lib_instance_host_var    std::unique_ptr<instance::region_host>    m_host_region  {nullptr}
	#define lib_instance_device_var  std::unique_ptr<instance::region_device>  m_device_region{nullptr}
#else
	#define lib_instance_host_type instance::iregion 
	#define lib_instance_device_type instance::iregion 
	#define lib_instance_host_var    
	#define lib_instance_device_var  
#endif

namespace core
{
	class LIBRARY_API system {
	public:
		system();
		~system();

		const lib_instance_host_type*   host_region() const;
		const lib_instance_device_type* device_region() const;

	private:
		lib_instance_host_var;
		lib_instance_device_var;
	};
}



