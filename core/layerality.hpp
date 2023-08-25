#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "cellularity.hpp"

#include "../deflib.inc"

namespace innate {
	struct LIBRARY_API layer {
		int width = -1;
		int height = -1;
		int spillovers_size = -1;
	};
}

namespace instance {
	/*struct readable_layer_innate {
		const innate::layer* layer;
		const std::vector<readable_cell_innate> cellularity;
	};

	class LIBRARY_API ilayerality {
	public:
		virtual const innate::layer& layer() const = 0;
		virtual ptree to_ptree() const = 0;
		virtual readable_cell_innate innate() const = 0;

		virtual __mem__ float* spillovers() const = 0;
		virtual size_t spillovers_szb() const = 0;

	};

	template<typename T, typename CELL> class layerality : public ilayerality {
	public:
		const T& innlayer() const;
		const innate::layer& layer() const override;

		__mem__ float* spillovers() const override;
		size_t spillovers_szb() const override;

		virtual ~layerality();

	protected:
		layerality(const innate::layer& layer);

		T m_innate = nullptr;

		__mem__ float* spillovers = nullptr;
		size_t m_spillovers_szb = 0;

		std::vector<std::unique_ptr<CELL>> m_cellularity;
	};*/
}

BOOST_HANA_ADAPT_STRUCT(innate::layer, width, height, spillovers_size);