#include "layerality_instance.hpp"
#include "assert.hpp"

#pragma warning(disable:6011)

namespace instance {
	template<typename CELL, typename SPLVR>
	const core::region& layerality<CELL, SPLVR>::region() const {
		return m_region;
	}

	template<typename CELL, typename SPLVR>
	readable_layer_instance layerality<CELL, SPLVR>::instance() const {
		std::vector<readable_cell_instance> cellularity;
		for (const auto& c : m_cellularitys)
			cellularity.push_back(c->instance());

		std::vector<readable_splvr_instance> spilloverity;
		for (const auto& splvr : m_spilloveritys)
			spilloverity.push_back(splvr->instance());

		return { spilloverity, cellularity };
	}

	template<typename CELL, typename SPLVR>
	layerality<CELL, SPLVR>::layerality(const core::region& r): m_region(r){
	}

	template<typename CELL, typename SPLVR>
	layerality<CELL, SPLVR>::~layerality() {
	}

	template<typename CELL, typename SPLVR>
	const std::unique_ptr<CELL>& layerality<CELL, SPLVR>::cellularity(int index) const {
		return m_cellularitys[index];
	}
	
	template<typename CELL, typename SPLVR>
	const std::unique_ptr<SPLVR>& layerality<CELL, SPLVR>::spilloverity(int index) const {
		return m_spilloveritys[index];
	}
}

namespace instance {

}