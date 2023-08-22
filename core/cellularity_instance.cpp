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
	std::unique_ptr<innate::cell> cellularity<T, TR>::to_innate(const ptree& root) {
		auto innate_cell_type
			= static_cast<innate::cluster::cluster_type>(root.get<int>("type"));

		std::unique_ptr<innate::cell> ptr(nullptr);
		cell_data_tuple::create_first(innate_cell_type, [&](auto p) {
			auto innate_extend_tree = root.get_child("innate_extend");
			boost::to(*p, innate_extend_tree);
			ptr = std::move(p);
			});

		if (!ptr.get())
			logexit();

		boost::to(*ptr, root);

		return ptr;
	}

	template<typename T, typename TR>
	ptree cellularity<T, TR>::to_ptree(innate::cell* c) {
		auto innate_c = boost::to_ptree(*c);
		cell_data_tuple::to_first(c, [&innate_c](auto* t) {
			innate_c.put_child("innate_extend", boost::to_ptree(*t));
			});
		return innate_c;
	}

	template<typename T, typename TR>
	cellularity<T, TR>::cellularity(const innate::layer& layer)
		: m_layer(layer) {
	}

	template<typename T, typename TR>
	size_t cellularity<T, TR>::calc_results_bytes(const innate::layer& layer) const {
		return layer.height * layer.width * sizeof(float);
	}

	template<typename T, typename TR>
	size_t cellularity<T, TR>::calc_cells_bytes(const innate::layer& layer, const innate::cell* c) const {
		if (!c)
			logexit();

		const auto size_types = cell_data_tuple::size(c);
		if (size_types.empty())
			logexit();

		if (size_types.size() < 2 || size_types.back() < 1)
			logexit();

		return layer.height * layer.width * size_types.back();
	}

	template<typename T, typename TR>
	cellularity<T, TR>::~cellularity() {
	};
}

namespace instance {
	host_cellularity::host_cellularity(const ptree& root, const innate::layer& layer)
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

	ptree host_cellularity::to_ptree() const
	{
		auto c = inncell().get();
		if (!c)
			logexit();

		return cellularity::to_ptree(c);
	}


	device_cellularity::device_cellularity(const ptree& root, const innate::layer& layer)
		: cellularity_gpu_type(layer)
	{
		auto innate = to_innate(root);

		auto c = innate.get();
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

	device_cellularity::~device_cellularity() {
		memory::remove_mempart(m_const_cell);
		memory::setup_const_memoryparts();

		if (m_results) cudaFree(m_results);
		if (m_cells) cudaFree(m_cells);
	}

	memory::const_empl::ptr device_cellularity::const_emplace_cell() const
	{
		return m_const_cell;
	}

	void device_cellularity::setup_const_memory(const innate::cell* c) {
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