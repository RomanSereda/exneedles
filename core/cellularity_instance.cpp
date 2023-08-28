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
	const innate::size& cellularity<T, TR>::size() const {
		return m_size;
	}

	template<typename T, typename TR>
	readable_cell_instance cellularity<T, TR>::instance() const {
		std::vector<readable_trmn_instance> terminality;
		for (const auto& trmn : m_terminalitys)
			terminality.push_back(trmn->instance());
		
		return {m_cells, m_results, m_cells_szb, m_results_szb, terminality};
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
	cellularity<T, TR>::cellularity(const innate::size& size)
		: m_size(size) {
	}

	template<typename T, typename TR>
	cellularity<T, TR>::~cellularity() {
	}
	template<typename T, typename TR>
	const std::unique_ptr<TR>& cellularity<T, TR>::terminality(int index) const {
		return m_terminalitys[index];
	}
}

namespace instance {
	cellularity_host::cellularity_host(const ptree& root, const innate::size& size)
		: cellularity_cpu_type(size){

		if (size.height < 1 || size.width < 1)
			logexit();

		m_innate = to_innate(root);
		if(!m_innate)
			logexit();

		m_results_szb = calc_results_bytes(size);
		m_cells_szb = calc_cells_bytes(size, m_innate.get());

		if (!m_results_szb || !m_cells_szb)
			logexit();

		m_results = (__mem__ float*)malloc(m_results_szb);
		m_cells = (__mem__ data::cell*)malloc(m_cells_szb);

		if (!m_cells || !m_results)
			logexit();

		memset(m_results, 0, m_results_szb);
		memset(m_cells, 0, m_cells_szb);

		for (const auto& child : boost::to_vector(root, "terminalitys")) {
			auto terminality = std::make_unique<terminality_host>(child, size);
			m_terminalitys.push_back(std::move(terminality));
		}
	}

	ptree cellularity_host::to_ptree() const {
		auto c = inncell().get();
		if (!c)
			logexit();

		auto root = cellularity::to_ptree(c);
		boost::add_array(root, "terminalitys", m_terminalitys);

		return root;
	}

	readable_cell_innate cellularity_host::innate() const {
		std::vector<readable_trmn_innate> terminality;
		for (const auto& trmn : m_terminalitys)
			terminality.push_back(trmn->innate());
		return { m_innate.get(), std::move(terminality) };
	}


	cellularity_device::cellularity_device(const ptree& root, const innate::size& size)
		: cellularity_gpu_type(size)
	{
		m_uptr_innate = to_innate(root);

		setup_const_memory();

		m_results_szb = calc_results_bytes(size);
		assert_err(cudaMalloc((void**)&m_results, m_results_szb));
		assert_err(cudaMemset((void*)m_results, 0, m_results_szb));

		m_cells_szb = calc_cells_bytes(size, m_uptr_innate.get());
		assert_err(cudaMalloc((void**)&m_cells, m_cells_szb));
		assert_err(cudaMemset((void*)m_cells, 0, m_cells_szb));

		if (!m_cells || !m_results)
			logexit();

		for (const auto& child : boost::to_vector(root, "terminalitys")) {
			auto terminality = std::make_unique<terminality_device>(child, size);
			m_terminalitys.push_back(std::move(terminality));
		}
	}

	cellularity_device::~cellularity_device() {
		memory::remove_mempart(m_const_cell);
		memory::setup_const_memoryparts();

		if (m_results) cudaFree(m_results);
		if (m_cells) cudaFree(m_cells);
	}

	ptree cellularity_device::to_ptree() const {
		auto root = icellularity::to_ptree(m_uptr_innate.get());
		boost::add_array(root, "terminalitys", m_terminalitys);
		return root;
	}

	readable_cell_innate cellularity_device::innate() const {
		std::vector<readable_trmn_innate> terminality;
		for (const auto& trmn : m_terminalitys)
			terminality.push_back(trmn->innate());

		return { m_uptr_innate.get(), terminality };
	}

	memory::const_empl::ptr cellularity_device::const_emplace_cell() const {
		return m_const_cell;
	}

	void cellularity_device::setup_const_memory() {
		auto c = m_uptr_innate.get();
		if (!c)
			logexit();

		cell_data_tuple::to_first(c, [&](auto* t0) {
			m_const_cell = memory::add_mempart(t0);
			});

		if (!m_const_cell)
			logexit();

		memory::setup_const_memoryparts();
		
		if (!m_const_cell->const_ptr)
			logexit();

		m_innate = (__const__ innate::cell**) &m_const_cell->const_ptr;
	}
}