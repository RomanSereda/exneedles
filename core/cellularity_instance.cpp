#include "cellularity_instance.hpp"
#include "layerality.hpp"

#include "memory.cuh"
#include "assert.hpp"

#pragma warning(disable:6011)

namespace instance {
	template<typename T, typename TR>
	const T& cellularity<T, TR>::inncell() const {
		return m_innate;
	}

	template<typename T, typename TR>
	const innate::layer& cellularity<T, TR>::layer() const {
		return m_layer;
	}

	template<typename T, typename TR>
	__mem__ float* cellularity<T, TR>::results() const {
		return m_results;
	}

	template<typename T, typename TR>
	__mem__ void* cellularity<T, TR>::cells() const {
		return m_cells;
	}

	template<typename T, typename TR>
	size_t cellularity<T, TR>::results_szb() const {
		return m_results_szb;
	}

	template<typename T, typename TR>
	size_t cellularity<T, TR>::cells_szb() const {
		return m_cells_szb;
	}

	template<typename T, typename TR>
	cellularity<T, TR>::cellularity(const innate::layer& layer)
		: m_layer(layer) {
	}

	template<typename T, typename TR>
	cellularity<T, TR>::~cellularity() {
	};
}

namespace instance {
	cellularity_host::cellularity_host(const ptree& root, const innate::layer& layer)
		: cellularity_cpu_type(layer){

		if (layer.height < 1 || layer.width < 1)
			logexit();

		m_innate = to_innate(root);
		if(!m_innate)
			logexit();

		m_results_szb = calc_results_bytes(layer);
		m_cells_szb = calc_cells_bytes(layer, m_innate.get());

		if (!m_results_szb || !m_cells_szb)
			logexit();

		m_results = (__mem__ float*)malloc(m_results_szb);
		m_cells = (__mem__ data::cell*)malloc(m_cells_szb);

		if (!m_cells || !m_results)
			logexit();

		memset(m_results, 0, m_results_szb);
		memset(m_cells, 0, m_cells_szb);
	}

	ptree cellularity_host::to_ptree() const {
		auto c = inncell().get();
		if (!c)
			logexit();

		return cellularity::to_ptree(c);
	}

	readable_cell_innate cellularity_host::innate() const {
		return { *m_innate.get() };
	}


	cellularity_device::cellularity_device(const ptree& root, const innate::layer& layer)
		: cellularity_gpu_type(layer)
	{
		m_uptr_innate = to_innate(root);

		auto c = m_uptr_innate.get();
		if (!c)
			logexit();

		cell_data_tuple::to_first(c, [&](auto* t0) {
			m_const_cell = memory::add_mempart(t0);
		});

		if (!m_const_cell)
			logexit();

		memory::setup_const_memoryparts();
		setup_const_memory(c);
	}

	cellularity_device::~cellularity_device() {
		memory::remove_mempart(m_const_cell);
		memory::setup_const_memoryparts();

		if (m_results) cudaFree(m_results);
		if (m_cells) cudaFree(m_cells);
	}

	ptree cellularity_device::to_ptree() const {
		return icellularity::to_ptree(m_uptr_innate.get());
	}

	readable_cell_innate cellularity_device::innate() const {
		return { *m_uptr_innate.get() };
	}

	memory::const_empl::ptr cellularity_device::const_emplace_cell() const {
		return m_const_cell;
	}

	void cellularity_device::setup_const_memory(const innate::cell* c) {
		if (!m_const_cell->const_ptr)
			logexit();

		m_innate = (__const__ innate::cell**) &m_const_cell->const_ptr;

		m_results_szb = calc_results_bytes(layer());
		assert_err(cudaMalloc((void**)&m_results, m_results_szb));
		assert_err(cudaMemset((void*)m_results, 0, m_results_szb));

		m_cells_szb = calc_cells_bytes(layer(), c);
		assert_err(cudaMalloc((void**)&m_cells, m_cells_szb));
		assert_err(cudaMemset((void*)m_cells, 0, m_cells_szb));

		if (!m_cells || !m_results)
			logexit();
	}
}