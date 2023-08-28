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
	layerality<CELL, SPLVR>::layerality(const innate::size& size): m_size(size){
		if (size.height < 1 || size.width < 1)
			logexit();
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
	
	layerality_host::layerality_host(const ptree& root, const innate::size& size): layerality_cpu_type(size) {
		for (const auto& child : boost::to_vector(root, "cellularity")) {
			auto cellularity = std::make_unique<cellularity_host>(child, size);
			m_cellularitys.push_back(std::move(cellularity));
		}

		for (const auto& child : boost::to_vector(root, "spilloverity")) {
			auto spilloverity = std::make_unique<spilloverity_host>(child, size);
			m_spilloveritys.push_back(std::move(spilloverity));
		}
	}
	
	layerality_device::layerality_device(const ptree& root, const innate::size& size): layerality_gpu_type(size) {
		for (const auto& child : boost::to_vector(root, "cellularity")) {
			auto cellularity = std::make_unique<cellularity_device>(child, size);
			m_cellularitys.push_back(std::move(cellularity));
		}

		for (const auto& child : boost::to_vector(root, "spilloverity")) {
			auto spilloverity = std::make_unique<spilloverity_device>(child, size);
			m_spilloveritys.push_back(std::move(spilloverity));
		}
	}
}

namespace instance {
	template<typename LR>
	region<LR>::region(const ptree& root) {
		boost::to(*((innate::size*)&m_size), root);
		if (m_size.height < 1 || m_size.width < 1)
			logexit();
	}

	template<typename LR>
	ptree region<LR>::to_ptree() const {
		ptree root;
		boost::add_array(root, "layeralitys", m_layeralitys);
		return root;
	}
	
	template<typename LR>
	const innate::size& region<LR>::size() const {
		return m_size;
	}
	
	template<typename LR>
	readable_region_innate region<LR>::innate() const {
		std::vector<readable_layer_innate> rg;
		for (const auto& l : m_layeralitys)
			rg.push_back(l->innate());

		return { rg };
	}
	
	template<typename LR>
	readable_region_instance region<LR>::instance() const {
		std::vector<readable_layer_instance> rg;
		for (const auto& l : m_layeralitys)
			rg.push_back(l->instance());

		return { rg };
	}
	
	template<typename LR>
	const std::unique_ptr<LR>& region<LR>::layerality(int index) const {
		return m_layeralitys[index];
	}

	region_host::region_host(const ptree& root): region_cpu_type(root) {
		for (const auto& child : boost::to_vector(root, "layeralitys")) {
			auto layerality = std::make_unique<layerality_host>(child, size());
			m_layeralitys.push_back(std::move(layerality));
		}
	}
	region_device::region_device(const ptree& root) : region_gpu_type(root) {
		for (const auto& child : boost::to_vector(root, "layeralitys")) {
			auto layerality = std::make_unique<layerality_device>(child, size());
			m_layeralitys.push_back(std::move(layerality));
		}
	}
}