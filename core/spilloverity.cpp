#include "spilloverity.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace innate {
	spillover::spillover(spillover_type t) : type(t) {
	}

	simple_spillover::simple_spillover() : spillover() {
	}

	void get_items(std::vector<spillover::spillover_type>& items) {
		std::vector spillover_types{ spillover::simple_spillover };

		items = spillover_types;
	}

	std::string to_string(spillover::spillover_type type) {
		switch (type)
		{
		case innate::spillover::simple_spillover:
			return "simple_spillover";
		default:
			break;
		}
		return "unknown type";
	}
}

namespace instance {
	std::unique_ptr<innate::spillover> ispilloverity::to_innate(const ptree& root, 
		                                                        innate::spillover::spillover_type deftype) 
	{
		if (root.empty() || root.find("innate_spillover") == root.not_found()) {
			std::unique_ptr<innate::spillover> ptr(nullptr);
			spillover_data_tuple::create_first(deftype, [&](auto p) {
				ptr = std::move(p);
			});

			console("warning: created spilloverity innate from default type");
			return std::move(ptr);
		}

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

	size_t ispilloverity::calc_spillovers_bytes(const innate::size& size, const innate::spillover* splvr) {
		if (!splvr)
			logexit();

		const size_t cell_size = (size_t)size.height * size.width;
		const auto size_types = spillover_data_tuple::size(splvr);
		if (size_types.size() < 2 || size_types.back() < 1)
			logexit();

		return cell_size * size_types.back();
	}
}



