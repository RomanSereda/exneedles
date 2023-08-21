#include "instance.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace instance {
	host_terminality::host_terminality(const ptree& root, const innate::layer& layer) : terminality(layer) {
		if (layer.height < 1 || layer.width < 1)
			logexit();

		*m_innate = to_innate(root);

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

	device_terminality::device_terminality(const ptree& root, const innate::layer& layer) : terminality(layer) {
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

	memory::const_empl::ptr device_terminality::const_emplace_cl() const
	{
		return m_const_cl;
	}

	memory::const_empl::ptr device_terminality::const_emplace_tr() const
	{
		return m_const_tr;
	}

	void device_terminality::setup_const_memory(innate::cluster* cl, innate::terminal* tr)
	{
		if (!m_const_cl->const_ptr || !m_const_tr->const_ptr)
			logexit();

		*m_innate = std::make_tuple((__const__ innate::cluster**) & m_const_cl->const_ptr,
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

namespace instance {
	host_celularity::host_celularity(const ptree& root, const innate::layer& layer)
		: UPTR_TEMPLATE_CELL(layer){

		if (layer.height < 1 || layer.width < 1)
			logexit();


	}

	ptree host_celularity::to_ptree() const
	{
		return ptree();
	}

	device_celularity::device_celularity(const ptree& root, const innate::layer& layer)
		: PTR_TEMPLATE_CELL(layer)
	{
	}

	device_celularity::~device_celularity()
	{
	}

	memory::const_empl::ptr device_celularity::const_emplace_cell() const
	{
		return memory::const_empl::ptr();
	}

	void device_celularity::setup_const_memory(const std::unique_ptr<innate::cell>& c)
	{
	}
}