#include "terminality.hpp"

namespace instance {
	ptree terminality::to_ptree() const {
		ptree root;
		
		if (auto cl = std::get<__const__ innate::cluster*>(innate)) {
			auto innate_cl = boost::to_ptree(*cl);
			cluster_tuple::foreach(cl, [&innate_cl](auto* t0) { innate_cl.put_child("innate_extend", boost::to_ptree(*t0)); });
			root.put_child("innate_cluster", innate_cl);
		}

		if (auto tr = std::get<__const__ innate::terminal*>(innate)) {
			auto innate_tr = boost::to_ptree(*tr);
			cluster_data_tuple::foreach(tr, [&innate_tr](auto* t0) { innate_tr.put_child("extend", boost::to_ptree(*t0)); });
			root.put_child("innate_terminal", innate_tr);
		}

		return root;
	}

}