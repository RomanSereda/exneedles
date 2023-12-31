#include "terminality.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace innate {
	terminal::terminal(terminal_type t) : type{ t }, sign{ positive } {};
	axon_simple::axon_simple() : terminal(terminal_type::axon_simple) {};
	synapse_simple::synapse_simple() : terminal(terminal_type::synapse_simple) {};
	
	cluster::cluster(cluster_type t) : type{ t } {};
	cluster_targeted::cluster_targeted() : cluster(cluster_type::cluster_targeted) {};

	void get_items(std::vector<terminal::terminal_type>& items) {
		std::vector terminal_types{ terminal::axon_simple, terminal::synapse_simple };
		items = terminal_types;
	}

	void get_items(std::vector<terminal::terminal_sign>& items) {
		std::vector terminal_signs{ terminal::positive, terminal::negative };
		items = terminal_signs;
	}

	void get_items(std::vector<cluster::cluster_type>& items) {
		std::vector cluster_types { cluster::cluster_targeted };
		items = cluster_types;
	}

	std::string to_string(terminal::terminal_type type) {
		switch (type)
		{
		case innate::terminal::axon_simple:
			return "axon_simple";
		case innate::terminal::synapse_simple:
			return "synapse_simple";
		default:
			break;
		}
		return "unknown type";
	}
	
	std::string to_string(terminal::terminal_sign type) {
		switch (type)
		{
		case innate::terminal::positive:
			return "positive";
		case innate::terminal::negative:
			return "negative";
		default:
			break;
		}
		return "unknown type";
	}

	std::string to_string(cluster::cluster_type type) {
		switch (type)
		{
		case innate::cluster::cluster_targeted:
			return "cluster_targeted";
		default:
			break;
		}
		return "unknown type";
	}
}

namespace instance {
	std::tuple<UPTR_TEMPLATE_TR> iterminality::to_innate(
		const ptree& root, const InnateTerminalityParam& def) 
	{
		auto cl = to_inncl(root, def.cl_type, def.width, def.height);
		auto tr = to_inntr(root, def.tr_type);

		return std::make_tuple<UPTR_TEMPLATE_TR>(std::move(cl), std::move(tr));
	}

	ptree iterminality::to_ptree(innate::cluster* cl, innate::terminal* tr) {
		if (!cl || !tr)
			logexit();

		ptree root;
		root.put_child("innate_cluster", to_ptree(cl));
		root.put_child("innate_terminal", to_ptree(tr));

		return root;
	}

	size_t iterminality::calc_results_bytes(const innate::size& size) {
		const size_t cell_size = size.height * size.width;
		const size_t size_bytes_results = cell_size * sizeof(float);

		return size_bytes_results;
	}

	size_t iterminality::calc_terminals_bytes(const innate::size& size,
		                                      const innate::cluster* cl, 
		                                      const innate::terminal* tr) {
		if (!cl || !tr)
			logexit();

		if (cl->height < 1 && cl->width < 1)
			logexit();

		const size_t terminals_per_cluster = cl->height * cl->width;
		const size_t cell_size = size.height * size.width;
		const auto size_types = cluster_data_tuple::size(tr);
		if (size_types.size() < 2 || size_types.back() < 1) logexit();
		const size_t bytes_per_cluster = size_types.back() * terminals_per_cluster;
		const size_t size_bytes_terminals = bytes_per_cluster * cell_size;

		return size_bytes_terminals;
	}

	ptree iterminality::to_ptree(innate::cluster* cl) {
		auto innate_cl = boost::to_ptree(*cl);
		cluster_tuple::to(cl, [&innate_cl](auto* t0) {
			innate_cl.put_child("innate_extend", boost::to_ptree(*t0));
		});
		return innate_cl;
	}

	ptree iterminality::to_ptree(innate::terminal* tr) {
		auto innate_tr = boost::to_ptree(*tr);
		cluster_data_tuple::to_first(tr, [&innate_tr](auto* t0) {
			innate_tr.put_child("innate_extend", boost::to_ptree(*t0));
		});
		return innate_tr;
	}

	std::unique_ptr<innate::cluster> iterminality::to_inncl(const ptree& root, 
		innate::cluster::cluster_type deftype, int width, int height)
	{
		if (root.empty() || root.find("innate_cluster") == root.not_found()) {
			std::unique_ptr<innate::cluster> ptr(nullptr);
			cluster_tuple::create(deftype, [&](auto p) {
				p->width = width;
				p->height = width;

				ptr = std::move(p);
			});

			console("warning: created cluster innate from default type");
			return std::move(ptr);
		}

		auto innate_cluster_tree = root.get_child("innate_cluster");
		auto innate_cluster_type
			= static_cast<innate::cluster::cluster_type>(innate_cluster_tree.get<int>("type"));

		std::unique_ptr<innate::cluster> ptr(nullptr);
		cluster_tuple::create(innate_cluster_type, [&](auto p) {
			auto innate_extend_tree = innate_cluster_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			ptr = std::move(p);
		});

		if (!ptr.get())
			logexit();

		boost::to(*ptr, innate_cluster_tree);

		return std::move(ptr);
	}

	std::unique_ptr<innate::terminal> iterminality::to_inntr(
		const ptree& root, innate::terminal::terminal_type deftype) 
	{
		if (root.empty() || root.find("innate_terminal") == root.not_found()) {
			std::unique_ptr<innate::terminal> ptr(nullptr);
			cluster_data_tuple::create_first(deftype, [&](auto p) {
				ptr = std::move(p);
			});

			console("warning: created terminal innate from default type");
			return std::move(ptr);
		}

		auto innate_terminal_tree = root.get_child("innate_terminal");
		auto innate_terminal_type
			= static_cast<innate::terminal::terminal_type>(innate_terminal_tree.get<int>("type"));

		std::unique_ptr<innate::terminal> ptr(nullptr);
		cluster_data_tuple::create_first(innate_terminal_type, [&](auto p) {
			auto innate_extend_tree = innate_terminal_tree.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			ptr = std::move(p);
		});

		if (!ptr.get())
			logexit();

		boost::to(*ptr, innate_terminal_tree);

		return std::move(ptr);
	}
}