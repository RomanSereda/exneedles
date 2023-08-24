#pragma once
#include "types.hpp"
#include "boost.hpp"
#include "memory.cuh"

#include "cellularity.hpp"
#include "terminality_instance.hpp"

namespace instance {
	template<typename T, typename TR> class cellularity: public icellularity {
	public:
		const T& inncell() const;

		const innate::layer& layer() const override;

		__mem__ float* results() const override;
		__mem__ void* cells() const override;

		size_t results_szb() const override;
		size_t cells_szb() const override;

		virtual ~cellularity();

	protected:
		cellularity(const innate::layer& layer);

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
	class cellularity_host : public cellularity_cpu_type {
	public:
		cellularity_host(const ptree& root, const innate::layer& layer);

		ptree to_ptree() const override;
		readable_cell_innate innate() const override;

	private:
		std::vector<std::unique_ptr<terminality_host>> m_terminalitys;
	};

	class cellularity_device : public cellularity_gpu_type {
	public:
		cellularity_device(const ptree& root, const innate::layer& layer);
		virtual ~cellularity_device();

		ptree to_ptree() const override;
		readable_cell_innate innate() const override;

		memory::const_empl::ptr const_emplace_cell() const;

	private:
		memory::const_empl::ptr m_const_cell = nullptr;
		std::unique_ptr<innate::cell> m_uptr_innate {nullptr};

		void setup_const_memory();

		std::vector<std::unique_ptr<terminality_device>> m_terminalitys;
	};

}
