#include "terminality.hpp"

namespace instance {
	terminality::terminality(const ptree& root) {

		innate::cluster* innate_cluster = nullptr;
		auto innate_cluster_tree = root.get_child("innate_cluster");
		auto innate_cluster_type = static_cast<innate::cluster::cluster_type>(std::stoi(innate_cluster_tree.get_value("type")));

		cluster_tuple::create(innate_cluster_type, [=](auto* p){
			auto innate_extend_tree = innate_cluster_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
		});


		auto innate_terminal_tree = root.get_child("innate_terminal");

		innate::terminal terminal;
		boost::to(terminal, innate_terminal_tree);

	}

	ptree terminality::to_ptree() const {
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

}