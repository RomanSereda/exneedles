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
		host = new instance::region_host(settings::instance().read());
		//device = new instance::region_device(host->to_ptree());
	}
	
	system::~system() {
		if (host) {
			settings::instance().save(host->to_ptree());

			delete host;
			host = nullptr;
		}

		/*if (device) {
			delete device;
			device = nullptr;
		}*/
	}

	instance::iregion& system::region() {
		return *host;
	}




}