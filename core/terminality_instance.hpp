#pragma once
#include "types.hpp"
#include "boost.hpp"
#include "memory.cuh"

#include "terminality.hpp"

namespace instance {
	template<typename CLST, typename TRMN> class terminality: public iterminality {
	public:
		const CLST& inncl() const;
		const TRMN& inntr() const;

		const core::region& region() const override;
		readable_trmn_instance instance() const override;

		__mem__ float* results() const override;
		__mem__ void* terminals() const override;

		size_t results_szb() const override;
		size_t terminals_szb() const override;

		virtual ~terminality();

	protected:
		terminality(const core::region& region);

		std::tuple<CLST, TRMN> m_innate {nullptr, nullptr};

		__mem__ float* m_results = nullptr; 
		__mem__ data::terminal* m_terminals = nullptr;

		size_t m_results_szb = 0;
		size_t m_terminals_szb = 0;
	private:
		const core::region& m_region;
	};

	using terminality_gpu_type = terminality<PTR_TEMPLATE_TR>;
	using terminality_cpu_type = terminality<UPTR_TEMPLATE_TR>;
}

namespace instance {
	class terminality_host : public terminality_cpu_type {
	public:
		terminality_host(const ptree& root, const core::region& region);
		ptree to_ptree() const override;

		readable_trmn_innate innate() const override;
	};

	class terminality_device : public terminality_gpu_type {
	public:
		terminality_device(const ptree& root, const core::region& region);
		virtual ~terminality_device();
		ptree to_ptree() const override;

		readable_trmn_innate innate() const override;

		memory::const_empl::ptr const_emplace_cl() const;
		memory::const_empl::ptr const_emplace_tr() const;

	private:
		memory::const_empl::ptr m_const_cl = nullptr;
		memory::const_empl::ptr m_const_tr = nullptr;
		std::tuple<UPTR_TEMPLATE_TR> m_uptr_innate {nullptr, nullptr};

		void setup_const_memory();
	};
}
