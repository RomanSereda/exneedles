#include "cellularity.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace innate {
	cell::cell(cell_type t): type(t){}
	cell_simple::cell_simple() : cell(cell_type::cell_simple) {};
	cell_exre::cell_exre() : cell(cell_type::cell_exre) {};
}

#pragma warning(disable:6011)

namespace instance {
	std::unique_ptr<innate::cell> icellularity::to_innate(const ptree& root) {
		auto innate_cell_type
			= static_cast<innate::cell::cell_type>(root.get<int>("type"));

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

	ptree icellularity::to_ptree(innate::cell* c) {
		auto innate_c = boost::to_ptree(*c);
		cell_data_tuple::to_first(c, [&innate_c](auto* t) {
			innate_c.put_child("innate_extend", boost::to_ptree(*t));
		});
		return innate_c;
	}

	size_t icellularity::calc_results_bytes(const innate::size& size) {
		return size.height * size.width * sizeof(float);
	}

	size_t icellularity::calc_cells_bytes(const innate::size& size, const innate::cell* c) {
		if (!c)
			logexit();

		const auto size_types = cell_data_tuple::size(c);
		if (size_types.empty())
			logexit();

		if (size_types.size() < 2 || size_types.back() < 1)
			logexit();

		return size.height * size.width * size_types.back();
	}
}