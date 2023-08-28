#include "layerality_instance.hpp"
#include "assert.hpp"

#pragma warning(disable:6011)

namespace instance {
	template<typename CELL, typename SPLVR>
	const innate::size& layerality<CELL, SPLVR>::size() const {
		return m_size;
	}

	template<typename CELL, typename SPLVR>
	readable_layer_innate layerality<CELL, SPLVR>::innate() const {
		std::vector<readable_cell_innate> cellularity;
		for (const auto& c : m_cellularitys)
			cellularity.push_back(c->innate());

		std::vector<readable_splvr_innate> spilloverity;
		for (const auto& splvr : m_spilloveritys)
			spilloverity.push_back(splvr->innate());

		return { spilloverity, cellularity };
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
	layerality<CELL, SPLVR>::layerality(const ptree& root, const innate::size& r): m_size(r){
		if (size.height < 1 || size.width < 1)
			logexit();

		for (const auto& child : boost::to_vector(root, "cellularity")) {
			auto cellularity = std::make_unique<cellularity_host>(child, size);
			m_cellularitys.push_back(std::move(cellularity));
		}

		for (const auto& child : boost::to_vector(root, "spilloverity")) {
			auto spilloverity = std::make_unique<spilloverity_host>(child, size);
			m_spilloveritys.push_back(std::move(spilloverity));
		}
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

	template<typename CELL, typename SPLVR>
	ptree layerality<CELL, SPLVR>::to_ptree() const {
		ptree root;
		boost::add_array(root, "cellularity", m_cellularitys);
		boost::add_array(root, "spilloverity", m_spilloveritys);

		return root;
	}
}