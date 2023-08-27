#include "layerality.hpp"
#include <memory>

namespace innate {
	layer::layer(const region& r):width(r.width), height(r.height) {
	}
}

namespace instance {
	std::unique_ptr<innate::layer> ilayerality::to_innate(const ptree& root, const innate::region& r) {
		std::unique_ptr<innate::layer> uptr{new innate::layer(r)};
		boost::to(*uptr, root);

		return std::move(uptr);
	}

	ptree ilayerality::to_ptree(innate::layer* l) {
		return boost::to_ptree(*l);
	}
}


