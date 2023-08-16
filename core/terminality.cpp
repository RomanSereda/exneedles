#include "terminality.hpp"
#include "layerality.hpp"

#pragma warning(disable:6011)

namespace instance {
	template<typename T0, typename T1> 
	size_t terminality<T0, T1>::calc_terminals_bytes(const innate::layer& layer,
		                                     const innate::cluster* cl, 
		                                     const innate::terminal* tr) const {
		if (!cl || !tr)
			logexit();

		if (cl->height < 1 && cl->width < 1)
			logexit();

		const size_t terminals_per_cluster = cl->height * cl->width;
		const size_t cell_size = layer.height * layer.width;
		const auto size_types = cluster_data_tuple::size(tr);
		if (size_types.size() < 2 || size_types.back() < 1) logexit();
		const size_t bytes_per_cluster = size_types.back() * terminals_per_cluster;
		const size_t size_bytes_terminals = bytes_per_cluster * cell_size;

		return size_bytes_terminals;
	}

	template<typename T0, typename T1>
	size_t terminality<T0, T1>::calc_results_bytes(const innate::layer& layer) const {
		const size_t cell_size = layer.height * layer.width;
		const size_t size_bytes_results = cell_size * sizeof(float);

		return size_bytes_results;
	}

	host_terminality::host_terminality(const ptree& root, const innate::layer& layer) {
		auto innate_cluster_tree = root.get_child("innate_cluster");
		auto innate_cluster_type 
			= static_cast<innate::cluster::cluster_type>(innate_cluster_tree.get<int>("type"));

		cluster_tuple::create(innate_cluster_type, [=](auto p){
			auto innate_extend_tree = innate_cluster_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			std::get<std::unique_ptr<innate::cluster>>(innate) = std::move(p);
		});

		auto innate_terminal_tree = root.get_child("innate_terminal");
		auto innate_terminal_type 
			= static_cast<innate::terminal::terminal_type>(innate_terminal_tree.get<int>("type"));

		cluster_data_tuple::create_first(innate_terminal_type, [=](auto p) {
			auto innate_extend_tree = innate_terminal_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			std::get<std::unique_ptr<innate::terminal>>(innate) = std::move(p);
		});

		if (layer.height < 1 || layer.width < 1)
			logexit();

		results   = (__mem__ float*)malloc(calc_results_bytes(layer));
		terminals = (__mem__ data::terminal*)malloc(calc_terminals_bytes(layer, inncl(), inntr()));

		if (!terminals || !results)
			logexit();
	}

	ptree host_terminality::to_ptree() const {
		ptree root;
		
		if (auto cl = inncl()) {
			auto innate_cl = boost::to_ptree(*cl);
			cluster_tuple::foreach(cl, [&innate_cl](auto* t0) { 
				innate_cl.put_child("innate_extend", boost::to_ptree(*t0));
			});
			
			root.put_child("innate_cluster", innate_cl);
		}

		if (auto tr = inntr()) {
			auto innate_tr = boost::to_ptree(*tr);
			cluster_data_tuple::foreach(tr, [&innate_tr](auto* t0) { 
				innate_tr.put_child("innate_extend", boost::to_ptree(*t0)); 
			});
			
			root.put_child("innate_terminal", innate_tr);
		}

		return root;
	}

	innate::cluster* host_terminality::inncl() const
	{
		return std::get<std::unique_ptr<innate::cluster>>(innate).get();
	}

	innate::terminal* host_terminality::inntr() const
	{
		return std::get<std::unique_ptr<innate::terminal>>(innate).get();
	}


}