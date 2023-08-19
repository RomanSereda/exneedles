#include "instance.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace instance {

	host_terminality::host_terminality(const ptree& root, const innate::layer& layer) : terminality() {
		*m_innate = std::make_tuple(to_inncl(root), to_inntr(root));

		if (layer.height < 1 || layer.width < 1)
			logexit();

		m_results_szb = calc_results_bytes(layer);
		m_terminals_szb = calc_terminals_bytes(layer, inncl().get(), inntr().get());

		m_results = (__mem__ float*)malloc(m_results_szb);
		m_terminals = (__mem__ data::terminal*)malloc(m_terminals_szb);

		if (!m_terminals || !m_results)
			logexit();

		memset(m_results, 0, m_results_szb);
		memset(m_terminals, 0, m_terminals_szb);
	}

	ptree host_terminality::to_ptree() const {
		ptree root;

		if (auto cl = inncl().get())
			root.put_child("innate_cluster", terminality::to_ptree(cl));

		if (auto tr = inntr().get())
			root.put_child("innate_terminal", terminality::to_ptree(tr));

		return root;
	}

	host_terminality::host_terminality()
	{
		innate::layer layer {128, 128, 1};

		ptree root;
		innate::cluster_targeted cl;
		cl.width = 8;
		cl.height = 8;
		cl.target_layer = 3;
		cl.target_region = 9;

		innate::synapse_simple tr;

		root.put_child("innate_cluster", terminality::to_ptree((innate::cluster*)&cl));
		root.put_child("innate_terminal", terminality::to_ptree((innate::terminal*)&tr));

		console(boost::to_string(root));

		host_terminality(root, layer);
	}

	device_terminality::device_terminality(const ptree& root, const innate::layer& layer) : terminality() {
		auto cl = to_inncl(root);
		auto tr = to_inntr(root);

		cluster_tuple::to(cl.get(), [&](auto* t0) {
			m_const_cl = memory::add_mempart(t0);
		});

		cluster_data_tuple::to_first(tr.get(), [&](auto* t0) {
			m_const_tr = memory::add_mempart(t0);
		});

		if (!m_const_cl || !m_const_tr)
			logexit();

		if (!m_const_cl->calc_const_ptr || !m_const_tr->calc_const_ptr)
			logexit();

		*m_innate = std::make_tuple((__const__ innate::cluster**) &m_const_cl->calc_const_ptr,
			                      (__const__ innate::terminal**) &m_const_tr->calc_const_ptr);

		m_results_szb = calc_results_bytes(layer);
		assert_err(cudaMalloc((void**)&m_results, m_results_szb));
		assert_err(cudaMemset((void*)m_results, 0, m_results_szb));

		m_terminals_szb = calc_terminals_bytes(layer, cl.get(), tr.get());
		assert_err(cudaMalloc((void**)&m_terminals, m_terminals_szb));
		assert_err(cudaMemset((void*)m_terminals, 0, m_terminals_szb));

		if (!m_terminals || !m_results)
			logexit();
	}

	device_terminality::~device_terminality() {
	}

}