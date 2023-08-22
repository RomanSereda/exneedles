#include "terminality_instance.hpp"
#include "layerality.hpp"

#include "memory.cuh"
#include "assert.hpp"

#pragma warning(disable:6011)

namespace instance {
	template<typename CLST, typename TRMN>
	terminality<CLST, TRMN>::terminality(const innate::layer& layer)
		: m_layer(layer) {
	}

	template<typename CLST, typename TRMN>
	terminality<CLST, TRMN>::~terminality() {
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
		return std::get<CLST>(m_innate);
	}

	template<typename CLST, typename TRMN>
	const TRMN& terminality<CLST, TRMN>::inntr() const {
		return std::get<TRMN>(m_innate);
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
	template<typename CLST, typename TRMN>
	std::tuple<UPTR_TEMPLATE_TR> terminality<CLST, TRMN>::to_innate(const ptree& root) {
		return std::make_tuple<UPTR_TEMPLATE_TR>(std::move(to_inncl(root)), std::move(to_inntr(root)));
	}

	template<typename CLST, typename TRMN>
	ptree terminality<CLST, TRMN>::to_ptree(innate::cluster* cl, innate::terminal* tr) {
		if (!cl || !tr)
			logexit();

		ptree root;
		root.put_child("innate_cluster", to_ptree(cl));
		root.put_child("innate_terminal", to_ptree(tr));

		return root;
	}
}


namespace instance {
	host_terminality::host_terminality(const ptree& root, const innate::layer& layer) 
		: terminality_cpu_type(layer) 
	{
		if (layer.height < 1 || layer.width < 1)
			logexit();

		m_innate = to_innate(root);

		if (!inncl().get() || !inntr().get())
			logexit();

		m_results_szb = calc_results_bytes(layer);
		m_terminals_szb = calc_terminals_bytes(layer, inncl().get(), inntr().get());

		if (!m_results_szb || !m_terminals_szb)
			logexit();

		m_results = (__mem__ float*)malloc(m_results_szb);
		m_terminals = (__mem__ data::terminal*)malloc(m_terminals_szb);

		if (!m_terminals || !m_results)
			logexit();

		memset(m_results, 0, m_results_szb);
		memset(m_terminals, 0, m_terminals_szb);
	}

	ptree host_terminality::to_ptree() const {
		ptree root;

		auto cl = inncl().get();
		auto tr = inntr().get();

		if (!cl || !tr)
			logexit();

		return terminality::to_ptree((innate::cluster*)cl, (innate::terminal*)tr);
	}

	device_terminality::device_terminality(const ptree& root, const innate::layer& layer) 
		: terminality_gpu_type(layer) 
	{
		auto innate = terminality::to_innate(root);

		auto cl = std::get<0>(innate).get();
		auto tr = std::get<1>(innate).get();

		cluster_tuple::to(cl, [&](auto* t0) {
			m_const_cl = memory::add_mempart(t0);
			});

		cluster_data_tuple::to_first(tr, [&](auto* t0) {
			m_const_tr = memory::add_mempart(t0);
			});

		if (!m_const_cl || !m_const_tr)
			logexit();

		memory::setup_const_memoryparts();
		setup_const_memory(cl, tr);
	}

	device_terminality::~device_terminality() {
		memory::remove_mempart(m_const_cl);
		memory::remove_mempart(m_const_tr);

		memory::setup_const_memoryparts();

		if (m_results) cudaFree(m_results);
		if (m_terminals) cudaFree(m_terminals);
	}

	memory::const_empl::ptr device_terminality::const_emplace_cl() const {
		return m_const_cl;
	}

	memory::const_empl::ptr device_terminality::const_emplace_tr() const {
		return m_const_tr;
	}

	void device_terminality::setup_const_memory(innate::cluster* cl, innate::terminal* tr) {
		if (!m_const_cl->const_ptr || !m_const_tr->const_ptr)
			logexit();

		m_innate = std::make_tuple((__const__ innate::cluster**) & m_const_cl->const_ptr,
			(__const__ innate::terminal**) & m_const_tr->const_ptr);

		m_results_szb = calc_results_bytes(layer());
		assert_err(cudaMalloc((void**)&m_results, m_results_szb));
		assert_err(cudaMemset((void*)m_results, 0, m_results_szb));

		m_terminals_szb = calc_terminals_bytes(layer(), cl, tr);
		assert_err(cudaMalloc((void**)&m_terminals, m_terminals_szb));
		assert_err(cudaMemset((void*)m_terminals, 0, m_terminals_szb));

		if (!m_terminals || !m_results)
			logexit();
	}
}
