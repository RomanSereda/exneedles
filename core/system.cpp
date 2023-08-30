#include "system.hpp"
#include "terminality.hpp"
#include "cellularity.hpp"
#include "layerality.hpp"
#include "terminality_instance.hpp"
#include "cellularity_instance.hpp"
#include "layerality_instance.hpp"

#include "settings.hpp"

namespace core {
	system::system() {
		device = new instance::region_device(settings::instance().read());
		host = new instance::region_host(device->to_ptree());
	}
	
	system::~system() {
		if (device) {
			settings::instance().save(device->to_ptree());
			delete device;
			device = nullptr;
		}
		if (host) {
			delete host;
			host = nullptr;
		}
	}




}