#pragma once
#include "types.hpp"
#include "celularity.hpp"

namespace innate {
	struct layer {
		int width = -1;
		int height = -1;
	};
}

namespace data {
	struct layer {
		innate::layer innate;
		std::vector<celular*> celulars;
		

	};
}