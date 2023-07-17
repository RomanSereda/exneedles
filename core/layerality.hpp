#pragma once
#include "types.hpp"
#include "celularity.hpp"

namespace innate {
	struct layer {
		int width = -1;
		int height = -1;
		int spillovers_count = 1;
		int celulars_count = -1;
	};

	struct region {
		int width = -1;
		int height = -1;
	};
}

namespace data {
	struct layer {
		__mem__ float* spillovers = nullptr;
	};

	struct region {
		std::vector<layer> layers;
	};

}