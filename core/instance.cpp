#include "instance.hpp"
#include "assert.hpp"
#include "layerality.hpp"

namespace instance {

	host_terminality::host_terminality(const ptree& root, const innate::layer& layer) : terminality() {
		*innate = std::make_tuple(to_inncl(root), to_inntr(root));

		if (layer.height < 1 || layer.width < 1)
			logexit();

		auto results_szb = calc_results_bytes(layer);
		auto terminals_szb = calc_terminals_bytes(layer, inncl().get(), inntr().get());

		results = (__mem__ float*)malloc(results_szb);
		terminals = (__mem__ data::terminal*)malloc(terminals_szb);

		memset(results, 0, results_szb);
		memset(terminals, 0, terminals_szb);

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

		*innate = std::make_tuple((__const__ innate::cluster**) &m_const_cl->calc_const_ptr,
			                      (__const__ innate::terminal**) &m_const_tr->calc_const_ptr);

		auto results_szb = calc_results_bytes(layer);
		assert_err(cudaMalloc((void**)&results, results_szb));
		assert_err(cudaMemset((void*)results, 0, results_szb));

		auto terminals_szb = calc_terminals_bytes(layer, cl.get(), tr.get());
		assert_err(cudaMalloc((void**)&terminals, terminals_szb));
		assert_err(cudaMemset((void*)terminals, 0, terminals_szb));

		if (!terminals || !results)
			logexit();
	}

	device_terminality::~device_terminality() {
	}

}