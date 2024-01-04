#include "layerality_instance.hpp"
#include "assert.hpp"

#pragma warning(disable:6011)

template<typename T1, typename T2> void rm(const std::string& id, T1& t1, T2& t2) {
	auto it = t1.find(id);
	if (it != t1.end()) {

		auto i = t2.begin();
		while (i != t2.end()) {
			if (it->second == i->get())
			{
				t1.erase(id);
				i = t2.erase(i);
				return;
			}
			else ++i;
		}
	}
	logexit();
}

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
		int id = 0;
		for (const auto& child : boost::to_vector(root, "cellularity")) {
			add_cell(std::to_string(id++), child, size);
		}

		id = 0;
		for (const auto& child : boost::to_vector(root, "spilloverity")) {
			add_splvr(std::to_string(id++), child, size);
		}
	}

	ilayerality& layerality_host::layerality() {
		return *this;
	}

	void layerality_host::rm_cell(const std::string& id) {
		rm(id, m_icellularitys, m_cellularitys);
	}

	void layerality_host::rm_splvr(const std::string& id) {
		rm(id, m_ispilloveritys, m_spilloveritys);
	}

	icellularity_host_accessor& layerality_host::add_cell(
		const std::string& id, innate::cell::cell_type deftype) 
	{
		return add_cell(id, ptree(), size(), deftype);
	}

	ispilloverity_host_accessor& layerality_host::add_splvr(
		const std::string& id, innate::spillover::spillover_type deftype) 
	{
		return add_splvr(id, ptree(), size(), deftype);
	}

	void layerality_host::get_cells(
		std::unordered_map<std::string, icellularity_host_accessor&>& cells) const
	{
		for (const auto& it : m_icellularitys)
			cells.emplace(it.first, *((icellularity_host_accessor*)it.second));
	}

	void layerality_host::get_splvrs(
		std::unordered_map<std::string, ispilloverity_host_accessor&>& splvrs) const
	{
		for (const auto& it : m_ispilloveritys)
			splvrs.emplace(it.first, *((ispilloverity_host_accessor*)it.second));
	}

	icellularity_host_accessor& layerality_host::add_cell(
		const std::string& id, const ptree& root, const innate::size& size, innate::cell::cell_type deftype)
	{
		auto cellularit = std::make_unique<cellularity_host>(root, size, deftype);
		auto ptr = cellularit.get();

		m_icellularitys.emplace(id, ptr);
		m_cellularitys.push_back(std::move(cellularit));

		return *ptr;
	}

	ispilloverity_host_accessor& layerality_host::add_splvr(
		const std::string& id, const ptree& root, const innate::size& size, innate::spillover::spillover_type deftype)
	{
		auto spilloverity = std::make_unique<spilloverity_host>(root, size, deftype);
		auto ptr = spilloverity.get();

		m_ispilloveritys.emplace(id, ptr);
		m_spilloveritys.push_back(std::move(spilloverity));

		return *ptr;
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
		ptree root = boost::to_ptree(m_size);
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

		return { m_size, rg };
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
		int id = 0;
		for (const auto& child : boost::to_vector(root, "layeralitys")) {
			add_layer(std::to_string(id++), child, size());
		}
	}

	iregion& region_host::region() {
		return *this;
	}

	void region_host::rm_layer(const std::string& id) {
		rm(id, m_ilayeralitys, m_layeralitys);
	}

	ilayerality_host_accessor& region_host::add_layer(const std::string& id) {
		return add_layer(id, ptree(), size());
	}

	void region_host::get_layers(
		std::unordered_map<std::string, ilayerality_host_accessor&>& layers) const 
	{
		for (const auto& it : m_ilayeralitys)
			layers.emplace(it.first, *((ilayerality_host_accessor*)it.second));
	}

	ilayerality_host_accessor& region_host::add_layer(
		const std::string& id, const ptree& root, const innate::size& size) 
	{
		auto layerality = std::make_unique<layerality_host>(root, size);
		auto ptr = layerality.get();

		m_ilayeralitys.emplace(id, ptr);
		m_layeralitys.push_back(std::move(layerality));

		return *ptr;
	}
	
	region_device::region_device(const ptree& root) : region_gpu_type(root) {
		for (const auto& child : boost::to_vector(root, "layeralitys")) {
			auto layerality = std::make_unique<layerality_device>(child, size());
			m_layeralitys.push_back(std::move(layerality));
		}
	}
}