#include "terminality.hpp"
#include "layerality.hpp"

#pragma warning(disable:6011)

namespace instance {
	host_terminality::host_terminality(const ptree& root, const innate::layer& layer) {

		innate::cluster* innate_cluster = nullptr;
		auto innate_cluster_tree = root.get_child("innate_cluster");
		auto innate_cluster_type = static_cast<innate::cluster::cluster_type>(innate_cluster_tree.get<int>("type"));

		cluster_tuple::create(innate_cluster_type, [=, &innate_cluster](auto* p){
			auto innate_extend_tree = innate_cluster_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			innate_cluster = p;
		});

		innate::terminal* innate_terminal = nullptr;
		auto innate_terminal_tree = root.get_child("innate_terminal");
		auto innate_terminal_type = static_cast<innate::terminal::terminal_type>(innate_terminal_tree.get<int>("type"));

		cluster_data_tuple::create_first(innate_terminal_type, [=, &innate_terminal](auto* p) {
			auto innate_extend_tree = innate_terminal_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			innate_terminal = p;
		});

		innate = std::make_tuple(innate_cluster, innate_terminal);

		if (layer.height < 1 || layer.width < 1)
			logexit();

		if (!innate_cluster || !innate_terminal)
			logexit();

		if (innate_cluster->height < 1 && innate_cluster->width < 1)
			logexit();
			
		const size_t terminals_per_cluster = innate_cluster->height * innate_cluster->width;
		const size_t cell_size = layer.height * layer.width;
		const auto size_types = cluster_data_tuple::size(innate_terminal);
		if (size_types.size() < 2 || size_types.back() < 1) logexit();
		const size_t bytes_per_cluster = size_types.back() * terminals_per_cluster;
		
		const size_t size_bytes_terminals = bytes_per_cluster * cell_size;
		terminals = (__mem__ data::terminal*)malloc(size_bytes_terminals);

		const size_t size_bytes_results = cell_size * sizeof(float);
		results = (__mem__ float*)malloc(size_bytes_results);

		if (!terminals || !results)
			logexit();
	}

	ptree host_terminality::to_ptree() const {
		ptree root;
		
		if (auto cl = std::get<__const__ innate::cluster*>(innate)) {
			auto innate_cl = boost::to_ptree(*cl);
			cluster_tuple::foreach(cl, [&innate_cl](auto* t0) { 
				innate_cl.put_child("innate_extend", boost::to_ptree(*t0)); 
			});
			
			root.put_child("innate_cluster", innate_cl);
		}

		if (auto tr = std::get<__const__ innate::terminal*>(innate)) {
			auto innate_tr = boost::to_ptree(*tr);
			cluster_data_tuple::foreach(tr, [&innate_tr](auto* t0) { 
				innate_tr.put_child("innate_extend", boost::to_ptree(*t0)); 
			});
			
			root.put_child("innate_terminal", innate_tr);
		}

		return root;
	}

	void* host_terminality::malloc(int size) const {
		return std::malloc(size);
	}

	void* device_terminality::malloc(int size) const {


		return nullptr;
	}
}