#pragma once
#include "types.hpp"

namespace innate {
	struct layer {
		int width = -1;
		int height = -1;
		int spillovers_count = 1;
	};
}

namespace data {
	struct layer {
		__const__ innate::layer* innate = nullptr;
		__mem__ float* spillovers = nullptr;
	};
}