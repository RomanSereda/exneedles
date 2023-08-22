#pragma once
#include "types.hpp"
#include "boost.hpp"
#include "memory.cuh"

#include "cellularity.hpp"
#include "terminality_instance.hpp"

namespace innate { struct layer; }
namespace instance {
	using cell_data_tuple = boost::spec_pair_tuple<std::tuple<innate::cell_simple, data::cell_simple>,
		std::tuple<innate::cell_exre, data::cell_exre>>;

	template<typename T, typename TR> class cellularity {
	public:
		const T& inncell() const;
		const innate::layer& layer() const;

		static std::unique_ptr<innate::cell> to_innate(const ptree& root);
		static ptree to_ptree(innate::cell* c);

		virtual ~cellularity();

	protected:
		cellularity(const innate::layer& layer);

		size_t calc_results_bytes(const innate::layer& layer) const;
		size_t calc_cells_bytes(const innate::layer& layer, const innate::cell* c) const;

		T m_innate = nullptr;                            

		__mem__ data::cell* m_cells = nullptr;
		__mem__ float* m_results = nullptr;

		size_t m_cells_szb = 0;
		size_t m_results_szb = 0;

		const innate::layer& m_layer;
	};

	using cellularity_gpu_type = cellularity<__const__ innate::cell**, terminality_gpu_type>;
	using cellularity_cpu_type = cellularity<std::unique_ptr<innate::cell>, terminality_cpu_type>;
}


namespace instance {
	class host_cellularity : public cellularity_cpu_type {
	public:
		host_cellularity(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;
	};

	class device_cellularity : public cellularity_gpu_type {
	public:
		device_cellularity(const ptree& root, const innate::layer& layer);
		virtual ~device_cellularity();

		memory::const_empl::ptr const_emplace_cell() const;

	private:
		memory::const_empl::ptr m_const_cell = nullptr;

		void setup_const_memory(const innate::cell* c);
	};

}
