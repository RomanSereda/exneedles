#pragma once
#include "types.hpp"
#include "boost.hpp"
#include "memory.cuh"

#include "terminality.hpp"

namespace innate { struct layer; }
namespace instance {
	using cluster_tuple = boost::spec_tuple<innate::cluster_targeted>;
	using cluster_data_tuple = boost::spec_pair_tuple<std::tuple<innate::axon_simple, data::axon_simple>,
		std::tuple<innate::synapse_simple, data::synapse_simple>>;

#define PTR_TEMPLATE_TR       __const__ innate::cluster**,      __const__ innate::terminal**
#define UPTR_TEMPLATE_TR std::unique_ptr<innate::cluster>, std::unique_ptr<innate::terminal>

	template<typename CLST, typename TRMN> class terminality {
	public:
		const CLST& inncl() const;
		const TRMN& inntr() const;

		const innate::layer& layer() const;

		static std::tuple<UPTR_TEMPLATE_TR> to_innate(const ptree& root);
		static ptree to_ptree(innate::cluster* cl, innate::terminal* tr);

		virtual ~terminality();

	protected:
		terminality(const innate::layer& layer);

		size_t calc_results_bytes(const innate::layer& layer) const;
		size_t calc_terminals_bytes(const innate::layer& layer,
			const innate::cluster* cl,
			const innate::terminal* tr) const;

		std::tuple<CLST, TRMN> m_innate {nullptr, nullptr};

		__mem__ float* m_results = nullptr; 
		__mem__ data::terminal* m_terminals = nullptr;

		size_t m_results_szb = 0;
		size_t m_terminals_szb = 0;

		const innate::layer& m_layer;

	private:
		static std::unique_ptr<innate::cluster> to_inncl(const ptree& root);
		static std::unique_ptr<innate::terminal> to_inntr(const ptree& root);

		static ptree to_ptree(innate::cluster* cl);
		static ptree to_ptree(innate::terminal* tr);
	};

	using terminality_gpu_type = terminality<PTR_TEMPLATE_TR>;
	using terminality_cpu_type = terminality<UPTR_TEMPLATE_TR>;
}

namespace instance {
	class host_terminality : public terminality_cpu_type {
	public:
		host_terminality(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;
	};

	class device_terminality : public terminality_gpu_type {
	public:
		device_terminality(const ptree& root, const innate::layer& layer);
		virtual ~device_terminality();

		memory::const_empl::ptr const_emplace_cl() const;
		memory::const_empl::ptr const_emplace_tr() const;

	private:
		memory::const_empl::ptr m_const_cl = nullptr;
		memory::const_empl::ptr m_const_tr = nullptr;

		void setup_const_memory(innate::cluster* cl, innate::terminal* tr);
	};
}
