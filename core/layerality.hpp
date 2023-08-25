#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "cellularity.hpp"

#include "../deflib.inc"

namespace innate {
	const int max_spillover_count = 4;

	struct LIBRARY_API spillover {
		enum spillover_type {
			simple_spillover
		} type;
	};

	struct LIBRARY_API layer {
		int width = -1;
		int height = -1;
	};

	struct LIBRARY_API region {

	};
}

namespace instance {
	struct readable_layer_innate {
		const innate::layer* layer;
		const std::vector<readable_cell_innate> cellularity;
	};
	struct readable_layer_instance {
		__mem__ float* spillovers = nullptr;
		size_t spillovers_szb = -1;
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

	protected:
		virtual __mem__ float* spillovers() const = 0;
		virtual size_t spillovers_szb() const = 0;

		static size_t calc_spillovers_bytes(const innate::layer& layer);
	};
}

BOOST_HANA_ADAPT_STRUCT(innate::spillover, type);
BOOST_HANA_ADAPT_STRUCT(innate::layer, width, height);