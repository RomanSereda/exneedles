#pragma once
#include "types.hpp"
#include "boost.hpp"
#include "memory.cuh"

#include "layerality.hpp"
#include "cellularity_instance.hpp"

namespace instance {
	/*template<typename T, typename CELL> class layerality : public ilayerality {
	public:
		const T& innlayer() const;
		const innate::region& region() const override;

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
