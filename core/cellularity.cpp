#include "cellularity.hpp"
#include "assert.hpp"

namespace innate {
	cell::cell(cell_type t): type(t){}
	cell_simple::cell_simple() : cell(cell_type::cell_simple) {};
	cell_exre::cell_exre() : cell(cell_type::cell_exre) {};
}

namespace instance {
	template<typename T> 
	const T& celularity<T>::inncell() const {
		return m_innate;
	}

	template<typename T> 
	const innate::layer& celularity<T>::layer() const {
		return m_layer;
	}

	template<typename T>
	std::unique_ptr<innate::cell> celularity<T>::to_inncell(const ptree& root)
	{
		auto innate_cell_type
			= static_cast<innate::cluster::cluster_type>(root.get<int>("type"));

		std::unique_ptr<innate::cell> ptr(nullptr);
		cell_tuple::create(innate_cell_type, [&](auto p) {
			auto innate_extend_tree = root.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			ptr = std::move(p);
		});

		if (!ptr.get())
			logexit();

		boost::to(*ptr, root);

		return ptr;
	}

	template<typename T>
	ptree celularity<T>::to_ptree(innate::cell* c)
	{
		auto innate_c = boost::to_ptree(*c);
		cell_tuple::to(c, [&innate_c](auto* t) {
			innate_c.put_child("innate_extend", boost::to_ptree(*t));
		});
		return innate_c;
	}

	template<typename T> 
	celularity<T>::celularity(const innate::layer& layer) : m_layer(layer) {}
	template<typename T> 
	celularity<T>::~celularity(){};


}