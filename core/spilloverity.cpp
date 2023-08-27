#include "spilloverity.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace innate {
	spillover::spillover(spillover_type t) : type(t) {
	}

	simple_spillover::simple_spillover() : spillover() {
	}
}

namespace instance {
	std::unique_ptr<innate::spillover> ispilloverity::to_innate(const ptree& root) {
		auto innate_spillover_tree = root.get_child("innate_spillover");
		auto innate_spillover_type
			= static_cast<innate::spillover::spillover_type>(innate_spillover_tree.get<int>("type"));

		std::unique_ptr<innate::spillover> ptr(nullptr);
		spillover_data_tuple::create_first(innate_spillover_type, [&](auto p) {
			auto innate_extend_tree = innate_spillover_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			ptr = std::move(p);
			});

		if (!ptr.get())
			logexit();

		boost::to(*ptr, innate_spillover_tree);

		return std::move(ptr);
	}

	ptree ispilloverity::to_ptree(innate::spillover* c) {
		auto innate_splvr = boost::to_ptree(*c);
		spillover_data_tuple::to_first(c, [&innate_splvr](auto* t0) {
			innate_splvr.put_child("innate_extend", boost::to_ptree(*t0));
			});
		return innate_splvr;
	}

	size_t ispilloverity::calc_spillovers_bytes(const innate::layer& layer, const innate::spillover* splvr) {
		if (!splvr)
			logexit();

		const size_t cell_size = layer.height * layer.width;
		const auto size_types = spillover_data_tuple::size(splvr);
		if (size_types.size() < 2 || size_types.back() < 1)
			logexit();

		return cell_size * size_types.back();
	}
}



