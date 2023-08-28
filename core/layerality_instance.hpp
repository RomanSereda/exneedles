#pragma once
#include "types.hpp"
#include "boost.hpp"
#include "memory.cuh"

#include "layerality.hpp"
#include "cellularity_instance.hpp"
#include "spilloverity_instance.hpp"

namespace instance {
	template<typename CELL, typename SPLVR> class layerality : public ilayerality {
	public:
		const core::region& region() const;

		readable_layer_instance instance() const override;
		layerality(const core::region& r);

		virtual ~layerality();

		const std::unique_ptr<CELL>& cellularity(int index) const;
		const std::unique_ptr<SPLVR>& spilloverity(int index) const;

	protected:
		std::vector<std::unique_ptr<CELL>> m_cellularitys;
		std::vector<std::unique_ptr<SPLVR>> m_spilloveritys;

	private:
		const core::region& m_region;
	};

	using layerality_gpu_type = layerality<cellularity_gpu_type, spilloverity_gpu_type>;
	using layerality_cpu_type = layerality<cellularity_cpu_type, spilloverity_cpu_type>;
}

namespace instance {
	class layerality_host : public layerality_cpu_type {
	public:
		layerality_host(const ptree& root, const core::region& r);
		ptree to_ptree() const override;

		readable_layer_innate innate() const override;
	};
}
