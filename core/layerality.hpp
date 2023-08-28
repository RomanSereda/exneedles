#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "cellularity.hpp"
#include "spilloverity.hpp"

#include "../deflib.inc"

namespace core {
	class LIBRARY_API region {
	public:
		region(const int width, const int height);

		int width;
		int height;

	private:
		static std::unique_ptr<region> to_innate(const ptree& root, int width, int height);
		static ptree to_ptree(region* r);
	};
}

namespace instance {
	struct readable_layer_innate {
		const std::vector<readable_splvr_innate> spillovers;
		const std::vector<readable_cell_innate> cellularity;
	};
	struct readable_layer_instance {
		const std::vector<readable_splvr_instance> spillovers;
		const std::vector<readable_cell_instance> cellularity;
	};

	class LIBRARY_API ilayerality {
	public:
		virtual const core::region& region() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_layer_innate innate() const = 0;
		virtual readable_layer_instance instance() const = 0;
	};
}