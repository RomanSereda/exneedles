#include "spilloverity_instance.hpp"
#include "spilloverity.hpp"
#include "layerality.hpp"

#include "memory.cuh"
#include "assert.hpp"

#pragma warning(disable:6011)

namespace instance {
	template<typename SPLVR>
	const SPLVR& spilloverity<SPLVR>::innsplvr() const {
		return m_innate;
	}

	template<typename SPLVR>
	const innate::size& spilloverity<SPLVR>::size() const {
		return m_size;
	}

	template<typename SPLVR>
	readable_splvr_instance spilloverity<SPLVR>::instance() const {
		return { m_spillovers, m_spillovers_szb };
	}

	template<typename SPLVR>
	__mem__ void* spilloverity<SPLVR>::spillovers() const {
		return m_spillovers;
	}

	template<typename SPLVR>
	size_t spilloverity<SPLVR>::spillovers_szb() const {
		return m_spillovers_szb;
	}

	template<typename SPLVR>
	spilloverity<SPLVR>::~spilloverity() {
	}

	template<typename SPLVR>
	spilloverity<SPLVR>::spilloverity(const innate::size& size) : m_size(size) {
	}
}

namespace instance {
	spilloverity_host::spilloverity_host(const ptree& root, const innate::size& size)
		: spilloverity_cpu_type(size) {
		if (size.height < 1 || size.width < 1)
			logexit();

		m_innate = to_innate(root);

		if (!innsplvr().get())
			logexit();

		m_spillovers_szb = calc_spillovers_bytes(size, innsplvr().get());

		if (!m_spillovers_szb)
			logexit();

		m_spillovers = (__mem__ data::spillover*)malloc(m_spillovers_szb);

		if (!m_spillovers )
			logexit();

		memset(m_spillovers, 0, m_spillovers_szb);
	}

	ptree spilloverity_host::to_ptree() const {
		ptree root;

		auto splvr = innsplvr().get();

		if (!splvr)
			logexit();

		return spilloverity::to_ptree((innate::spillover*)splvr);
	}

	readable_splvr_innate spilloverity_host::innate() const {
		auto splvr = innsplvr().get();

		if (!splvr)
			logexit();

		return { splvr };
	}


	spilloverity_device::spilloverity_device(const ptree& root, const innate::size& size)
		: spilloverity_gpu_type(size) {
	
		m_uptr_innate = spilloverity::to_innate(root);

		setup_const_memory();

		m_spillovers_szb = calc_spillovers_bytes(size, m_uptr_innate.get());
		assert_err(cudaMalloc((void**)&m_spillovers, m_spillovers_szb));
		assert_err(cudaMemset((void*)m_spillovers, 0, m_spillovers_szb));

		if (!m_spillovers)
			logexit();
	}

	spilloverity_device::~spilloverity_device() {
		memory::remove_mempart(m_const_spillover);
		memory::setup_const_memoryparts();

		if (m_spillovers) cudaFree(m_spillovers);
	}

	ptree spilloverity_device::to_ptree() const {
		auto splvr = m_uptr_innate.get();
		if (!splvr)
			logexit();

		return spilloverity::to_ptree(splvr);
	}

	readable_splvr_innate spilloverity_device::innate() const {
		auto splvr = m_uptr_innate.get();
		if (!splvr)
			logexit();

		return { splvr };
	}

	memory::const_empl::ptr spilloverity_device::const_emplace_spillover() const {
		return m_const_spillover;
	}

	void spilloverity_device::setup_const_memory() {
		auto splvr = m_uptr_innate.get();
		if (!splvr)
			logexit();

		spillover_data_tuple::to_first(splvr, [&](auto* t0) {
			m_const_spillover = memory::add_mempart(t0);
		});

		if (!m_const_spillover)
			logexit();

		memory::setup_const_memoryparts();

		if (!m_const_spillover->const_ptr)
			logexit();

		m_innate = (__const__ innate::spillover**) &m_const_spillover->const_ptr;
	}
}