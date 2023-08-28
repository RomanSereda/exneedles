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

		const innate::size& size() const override;
		readable_cell_instance instance() const override;

		__mem__ float* results() const override;
		__mem__ void* cells() const override;

		size_t results_szb() const override;
		size_t cells_szb() const override;

		virtual ~cellularity();

		const std::unique_ptr<TR>& terminality(int index) const;

	protected:
		cellularity(const innate::size& size);

		T m_innate = nullptr;                            

		__mem__ data::cell* m_cells = nullptr;
		__mem__ float* m_results = nullptr;

		size_t m_cells_szb = 0;
		size_t m_results_szb = 0;

		std::vector<std::unique_ptr<TR>> m_terminalitys;

	private:
		const innate::size& m_size;
	};

	using cellularity_gpu_type = cellularity<__const__ innate::cell**, terminality_gpu_type>;
	using cellularity_cpu_type = cellularity<std::unique_ptr<innate::cell>, terminality_cpu_type>;
}


namespace instance {
	class cellularity_host : public cellularity_cpu_type {
	public:
		cellularity_host(const ptree& root, const innate::size& size);
		ptree to_ptree() const override;

		readable_cell_innate innate() const override;
	};

	class cellularity_device : public cellularity_gpu_type {
	public:
		cellularity_device(const ptree& root, const innate::size& size);
		virtual ~cellularity_device();
		ptree to_ptree() const override;

		readable_cell_innate innate() const override;

		memory::const_empl::ptr const_emplace_cell() const;

	private:
		memory::const_empl::ptr m_const_cell = nullptr;
		std::unique_ptr<innate::cell> m_uptr_innate {nullptr};

		void setup_const_memory();
	};
}
