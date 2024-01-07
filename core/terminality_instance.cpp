#include "terminality_instance.hpp"
#include "layerality.hpp"

#include "memory.cuh"
#include "assert.hpp"

#pragma warning(disable:6011)

namespace instance {
	template<typename CLST, typename TRMN>
	terminality<CLST, TRMN>::terminality(const innate::size& size)
		: m_size(size) {
	}

	template<typename CLST, typename TRMN>
	terminality<CLST, TRMN>::~terminality() {
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
	const innate::size& terminality<CLST, TRMN>::size() const {
		return m_size;
	}

	template<typename CLST, typename TRMN>
	readable_trmn_instance terminality<CLST, TRMN>::instance() const
	{
		return {m_terminals, m_results, m_terminals_szb, m_results_szb};
	}

	template<typename CLST, typename TRMN>
	__mem__ float* terminality<CLST, TRMN>::results() const {
		return m_results;
	}

	template<typename CLST, typename TRMN>
	__mem__ void* terminality<CLST, TRMN>::terminals() const {
		return m_terminals;
	}
	
	template<typename CLST, typename TRMN>
	size_t terminality<CLST, TRMN>::results_szb() const {
		return m_results_szb;
	}
	
	template<typename CLST, typename TRMN>
	size_t terminality<CLST, TRMN>::terminals_szb() const {
		return m_terminals_szb;
	}
}


namespace instance {
	terminality_host::terminality_host(const ptree& root, 
		                               const innate::size& size, 
		                               const InnateTerminalityParam& def)
		: terminality_cpu_type(size)
	{
		if (size.height < 1 || size.width < 1)
			logexit();

		m_innate = to_innate(root, def);

		if (!inncl().get() || !inntr().get())
			logexit();

		m_results_szb = calc_results_bytes(size);
		m_terminals_szb = calc_terminals_bytes(size, inncl().get(), inntr().get());

		if (!m_results_szb || !m_terminals_szb)
			logexit();

		m_results = (__mem__ float*)malloc(m_results_szb);
		m_terminals = (__mem__ data::terminal*)malloc(m_terminals_szb);

		if (!m_terminals || !m_results)
			logexit();

		memset(m_results, 0, m_results_szb);
		memset(m_terminals, 0, m_terminals_szb);
	}

	ptree terminality_host::to_ptree() const {
		ptree root;

		auto cl = inncl().get();
		auto tr = inntr().get();

		if (!cl || !tr)
			logexit();

		return terminality::to_ptree((innate::cluster*)cl, (innate::terminal*)tr);
	}

	iterminality& terminality_host::terminality() {
		return *this;
	}

	readable_trmn_innate terminality_host::innate() const {
		auto cl = inncl().get();
		auto tr = inntr().get();

		if (!cl || !tr)
			logexit();

		return {cl, tr};
	}

	terminality_device::terminality_device(const ptree& root, const innate::size& size)
		: terminality_gpu_type(size)
	{
		m_uptr_innate = terminality::to_innate(root);

		setup_const_memory();

		m_results_szb = calc_results_bytes(size);
		assert_err(cudaMalloc((void**)&m_results, m_results_szb));
		assert_err(cudaMemset((void*)m_results, 0, m_results_szb));

		m_terminals_szb = calc_terminals_bytes(size, std::get<0>(m_uptr_innate).get(),
			                                          std::get<1>(m_uptr_innate).get());
		assert_err(cudaMalloc((void**)&m_terminals, m_terminals_szb));
		assert_err(cudaMemset((void*)m_terminals, 0, m_terminals_szb));

		if (!m_terminals || !m_results)
			logexit();
	}

	ptree terminality_device::to_ptree() const {
		auto cl = std::get<0>(m_uptr_innate).get();
		auto tr = std::get<1>(m_uptr_innate).get();

		if (!cl || !tr)
			logexit();

		return iterminality::to_ptree(cl, tr);
	}

	readable_trmn_innate terminality_device::innate() const {
		auto cl = std::get<0>(m_uptr_innate).get();
		auto tr = std::get<1>(m_uptr_innate).get();

		if (!cl || !tr)
			logexit();

		return { cl , tr};
	}

	terminality_device::~terminality_device() {
		memory::remove_mempart(m_const_cl);
		memory::remove_mempart(m_const_tr);

		memory::setup_const_memoryparts();

		if (m_results) cudaFree(m_results);
		if (m_terminals) cudaFree(m_terminals);
	}

	memory::const_empl::ptr terminality_device::const_emplace_cl() const {
		return m_const_cl;
	}

	memory::const_empl::ptr terminality_device::const_emplace_tr() const {
		return m_const_tr;
	}

	void terminality_device::setup_const_memory() {
		auto cl = std::get<0>(m_uptr_innate).get();
		auto tr = std::get<1>(m_uptr_innate).get();

		cluster_tuple::to(cl, [&](auto* t0) {
			m_const_cl = memory::add_mempart(t0);
		});

		cluster_data_tuple::to_first(tr, [&](auto* t0) {
			m_const_tr = memory::add_mempart(t0);
		});

		memory::setup_const_memoryparts();

		if (!m_const_cl || !m_const_tr)
			logexit();
		
		if (!m_const_cl->const_ptr || !m_const_tr->const_ptr)
			logexit();

		m_innate = std::make_tuple((__const__ innate::cluster**) & m_const_cl->const_ptr,
			(__const__ innate::terminal**) & m_const_tr->const_ptr);
	}
}
