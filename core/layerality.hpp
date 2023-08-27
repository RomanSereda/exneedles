#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "cellularity.hpp"
#include "spilloverity.hpp"

#include "../deflib.inc"

namespace innate {
	struct LIBRARY_API layer {
		int width = -1;
		int height = -1;
	};

	struct LIBRARY_API region {
		int width = -1;
		int height = -1;
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
		virtual const innate::region& region() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_layer_innate innate() const = 0;
		virtual readable_layer_instance instance() const = 0;

		static std::unique_ptr<innate::layer> to_innate(const ptree& root);
		static ptree to_ptree(innate::layer* l);
	};
}

BOOST_HANA_ADAPT_STRUCT(innate::layer, width, height);