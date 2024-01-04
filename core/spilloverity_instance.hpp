#pragma once
#pragma once
#include "types.hpp"
#include "boost.hpp"
#include "memory.cuh"

#include "spilloverity.hpp"

namespace instance {
	template<typename SPLVR> class spilloverity : public ispilloverity {
	public:
		const SPLVR& innsplvr() const;

		const innate::size& size() const override;
		readable_splvr_instance instance() const override;

		__mem__ void* spillovers() const override;
		size_t spillovers_szb() const override;

		virtual ~spilloverity();

	protected:
		spilloverity(const innate::size& size);

		SPLVR m_innate = nullptr;

		__mem__ data::spillover* m_spillovers = nullptr;
		size_t m_spillovers_szb = 0;

	private:
		const innate::size& m_size;
	};

	using spilloverity_gpu_type = spilloverity<__const__ innate::spillover**>;
	using spilloverity_cpu_type = spilloverity<std::unique_ptr<innate::spillover>>;
}

namespace instance {
	class spilloverity_host : public spilloverity_cpu_type, public ispilloverity_host_accessor {
	public:
		spilloverity_host(const ptree& root, const innate::size& size);
		ptree to_ptree() const override;

		readable_splvr_innate innate() const override;
	};

	class spilloverity_device : public spilloverity_gpu_type {
	public:
		spilloverity_device(const ptree& root, const innate::size& size);
		virtual ~spilloverity_device();
		ptree to_ptree() const override;

		readable_splvr_innate innate() const override;

		memory::const_empl::ptr const_emplace_spillover() const;

	private:
		memory::const_empl::ptr m_const_spillover = nullptr;
		std::unique_ptr<innate::spillover> m_uptr_innate {nullptr};

		void setup_const_memory();
	};
}