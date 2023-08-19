#include "instance.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace instance {

	host_terminality::host_terminality(const ptree& root, const innate::layer& layer) : terminality() {
		*innate = std::make_tuple(to_inncl(root), to_inntr(root));

		if (layer.height < 1 || layer.width < 1)
			logexit();

		results = (__mem__ float*)malloc(calc_results_bytes(layer));
		terminals = (__mem__ data::terminal*)malloc(calc_terminals_bytes(layer, inncl().get(), inntr().get()));

		if (!terminals || !results)
			logexit();
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

		device_terminality(root, layer);
	}

	device_terminality::device_terminality(const ptree& root, const innate::layer& layer) : terminality() {
		auto cl = to_inncl(root);
		auto tr = to_inntr(root);

		cluster_tuple::to(cl.get(), [&](auto* t0) {
			m_dcm_cl = dev_const_mem::add_mempart(t0);
		});

		cluster_data_tuple::to_first(tr.get(), [&](auto* t0) {
			m_dcm_tr = dev_const_mem::add_mempart(t0);
		});

		if (!m_dcm_cl || !m_dcm_tr)
			logexit();

		if (!m_dcm_cl->p || !m_dcm_tr->p)
			logexit();

		*innate = std::make_tuple((__const__ innate::cluster**) & m_dcm_cl->p,
			                      (__const__ innate::terminal**) & m_dcm_tr->p);

		auto results_szb = calc_results_bytes(layer);
		assert_err(cudaMalloc((void**)&results, results_szb));

		auto terminals_szb = calc_terminals_bytes(layer, cl.get(), tr.get());
		assert_err(cudaMalloc((void**)&terminals, terminals_szb));

		if (!terminals || !results)
			logexit();
	}

	device_terminality::~device_terminality() {
	}

}