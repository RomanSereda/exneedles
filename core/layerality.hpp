#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "cellularity.hpp"
#include "spilloverity.hpp"

#include "../deflib.inc"

namespace innate {
	struct size {
		int width;
		int height;
	};
}

BOOST_HANA_ADAPT_STRUCT(innate::size, width, height);

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
		virtual const innate::size& size() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_layer_innate innate() const = 0;
		virtual readable_layer_instance instance() const = 0;
	};

	struct readable_region_innate {
		const std::vector<readable_layer_innate> layer;
	};
	struct readable_region_instance {
		const std::vector<readable_layer_instance> layer;
	};

	class LIBRARY_API iregion {
	public:
		virtual const innate::size& size() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_region_innate innate() const = 0;
		virtual readable_region_instance instance() const = 0;
	};
}