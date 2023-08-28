#include "layerality.hpp"
#include <memory>

namespace core {
	region::region(const int width, const int height)
		: width(width), height(height) {
	}

	std::unique_ptr<region> region::to_innate(const ptree& root, int width, int height) {
		std::unique_ptr<region> uptr{new region(width, height)};
		

		/*-----------------------*/

		return std::move(uptr);
	}

	ptree region::to_ptree(region* r) {
		ptree root;

		/*-----------------------*/

		return root;
	}
}





