#include "terminality.hpp"
#include "layerality.hpp"

#include "memory.cuh"
#include "assert.hpp"

#pragma warning(disable:6011)

namespace innate {
	terminal::terminal(terminal_type t) : type{ t } {};
	axon_simple::axon_simple() : terminal(terminal_type::axon_simple) {};
	synapse_simple::synapse_simple() : terminal(terminal_type::synapse_simple), sign{ positive } {};
	
	cluster::cluster(cluster_type t) : type{ t } {};
	cluster_targeted::cluster_targeted() : cluster(cluster_type::cluster_targeted) {};
}

namespace instance {
	template<typename CLST, typename TRMN>
	terminality<CLST, TRMN>::terminality(const innate::layer& layer): m_layer(layer){
		m_innate = new std::tuple<CLST, TRMN>();
	}
	template<typename CLST, typename TRMN>
	terminality<CLST, TRMN>::~terminality() {
		delete m_innate;
	}

	template<typename CLST, typename TRMN>
	size_t terminality<CLST, TRMN>::calc_terminals_bytes(const innate::layer& layer,
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

	template<typename CLST, typename TRMN>
	size_t terminality<CLST, TRMN>::calc_results_bytes(const innate::layer& layer) const {
		const size_t cell_size = layer.height * layer.width;
		const size_t size_bytes_results = cell_size * sizeof(float);

		return size_bytes_results;
	}

	template<typename CLST, typename TRMN>
	const CLST& terminality<CLST, TRMN>::inncl() const {
		return std::get<CLST>(*m_innate);
	}

	template<typename CLST, typename TRMN>
	const TRMN& terminality<CLST, TRMN>::inntr() const {
		return std::get<TRMN>(*m_innate);
	}

	template<typename CLST, typename TRMN>
	const innate::layer& terminality<CLST, TRMN>::layer() const
	{
		return m_layer;
	}

	template<typename CLST, typename TRMN>
	ptree terminality<CLST, TRMN>::to_ptree(innate::cluster* cl) {
		auto innate_cl = boost::to_ptree(*cl);
		cluster_tuple::to(cl, [&innate_cl](auto* t0) {
			innate_cl.put_child("innate_extend", boost::to_ptree(*t0));
		});
		return innate_cl;
	}

	template<typename CLST, typename TRMN>
	ptree terminality<CLST, TRMN>::to_ptree(innate::terminal* tr) {
		auto innate_tr = boost::to_ptree(*tr);
		cluster_data_tuple::to_first(tr, [&innate_tr](auto* t0) {
			innate_tr.put_child("innate_extend", boost::to_ptree(*t0));
		});
		return innate_tr;
	}

	template<typename CLST, typename TRMN>
	std::unique_ptr<innate::cluster> terminality<CLST, TRMN>::to_inncl(const ptree& root) {
		boost::to_string(root);
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

	template<typename CLST, typename TRMN>
	std::unique_ptr<innate::terminal> terminality<CLST, TRMN>::to_inntr(const ptree& root) {
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