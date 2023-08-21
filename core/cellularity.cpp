#include "cellularity.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace innate {
	cell::cell(cell_type t): type(t){}
	cell_simple::cell_simple() : cell(cell_type::cell_simple) {};
	cell_exre::cell_exre() : cell(cell_type::cell_exre) {};
}

namespace instance {
	template<typename T, typename TR>
	const T& cellularity<T, TR>::inncell() const {
		return m_innate;
	}

	template<typename T, typename TR>
	const innate::layer& cellularity<T, TR>::layer() const {
		return m_layer;
	}

	template<typename T, typename TR>
	std::unique_ptr<innate::cell> cellularity<T, TR>::to_innate(const ptree& root) {
		auto innate_cell_type
			= static_cast<innate::cluster::cluster_type>(root.get<int>("type"));

		std::unique_ptr<innate::cell> ptr(nullptr);
		cell_data_tuple::create_first(innate_cell_type, [&](auto p) {
			auto innate_extend_tree = root.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			ptr = std::move(p);
		});

		if (!ptr.get())
			logexit();

		boost::to(*ptr, root);

		return ptr;
	}

	template<typename T, typename TR>
	ptree cellularity<T, TR>::to_ptree(innate::cell* c) {
		auto innate_c = boost::to_ptree(*c);
		cell_data_tuple::to_first(c, [&innate_c](auto* t) {
			innate_c.put_child("innate_extend", boost::to_ptree(*t));
		});
		return innate_c;
	}

	template<typename T, typename TR>
	const std::vector<std::unique_ptr<TR>>& cellularity<T, TR>::terminalitys() const {
		return *m_terminalitys;
	}

	template<typename T, typename TR>
	cellularity<T, TR>::cellularity(const innate::layer& layer) 
		: m_layer(layer), m_terminalitys(new std::vector<std::unique_ptr<TR>>()){}

	template<typename T, typename TR>
	size_t cellularity<T, TR>::calc_results_bytes(const innate::layer& layer) const {
		return layer.height * layer.width * sizeof(float);
	}

	template<typename T, typename TR>
	size_t cellularity<T, TR>::calc_cells_bytes(const innate::layer& layer, const innate::cell* c) const {
		if (!c)
			logexit();

		const auto size_types = cell_data_tuple::size(c);
		if (size_types.empty())
			logexit();

		if (size_types.size() < 2 || size_types.back() < 1)
			logexit();

		return layer.height * layer.width * size_types.back();
	}

	template<typename T, typename TR>
	cellularity<T, TR>::~cellularity(){
		if (m_terminalitys) delete m_terminalitys;
	};


}