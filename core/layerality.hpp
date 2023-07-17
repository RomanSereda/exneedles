#pragma once
#include "types.hpp"
#include "celularity.hpp"

namespace innate {
	struct layer {
		int width = -1;
		int height = -1;
		int spillovers_count = 1;
	};

	struct region {
		int width = -1;
		int height = -1;
	};
}

namespace data {
	struct layer {
		innate::layer innate;
		std::vector<celular*> celulars;
		std::vector<__mem__ float*> spillover;
	};

	struct region {
		innate::region innate;
		std::vector<layer> layers;
	};

}