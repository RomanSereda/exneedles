#include "terminality.hpp"

namespace instance {
	ptree cluster::to_ptree() const {
		ptree root;
		auto cl = std::get<__const__ innate::cluster*>(innate);
		auto tr = std::get<__const__ innate::terminal*>(innate);

		if (!cl) cl = new innate::cluster(innate::cluster::cluster_targeted);
		if (!tr) tr = new innate::terminal(innate::terminal::synapse_simple);

		root.put_child("innate_cluster", boost::to_ptree(*cl));
		root.put_child("innate_terminal", boost::to_ptree(*tr));

		return root;
	}

}