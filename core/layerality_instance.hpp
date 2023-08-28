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
		const innate::size& size() const;

		readable_layer_innate innate() const override;
		readable_layer_instance instance() const override;

		virtual ~layerality();

		const std::unique_ptr<CELL>& cellularity(int index) const;
		const std::unique_ptr<SPLVR>& spilloverity(int index) const;

		ptree to_ptree() const override;

	protected:
		layerality(const innate::size& size);

		std::vector<std::unique_ptr<CELL>> m_cellularitys;
		std::vector<std::unique_ptr<SPLVR>> m_spilloveritys;

	private:
		const innate::size& m_size;
	};

	using layerality_cpu_type = layerality<cellularity_cpu_type, spilloverity_cpu_type>;
	using layerality_gpu_type = layerality<cellularity_gpu_type, spilloverity_gpu_type>;

	class layerality_host : public layerality_cpu_type {
	public:
		layerality_host(const ptree& root, const innate::size& size);
	};

	class layerality_device : public layerality_gpu_type {
	public:
		layerality_device(const ptree& root, const innate::size& size);
	};
}

namespace instance {
	template<typename LR> class region : public iregion {
	public:
		region(const ptree& root);
		ptree to_ptree() const override;

		const innate::size& size() const override;

		readable_region_innate innate() const override;
		readable_region_instance instance() const override;

		const std::unique_ptr<LR>& layerality(int index) const;

	protected:
		const innate::size m_size;
		std::vector<std::unique_ptr<LR>> m_layeralitys;
	};

	using region_cpu_type = region<layerality_cpu_type>;
	using region_gpu_type = region<layerality_gpu_type>;

	class region_host : public region_cpu_type {
	public:
		region_host(const ptree& root);
	};

	class region_device : public region_gpu_type {
	public:
		region_device(const ptree& root);
	};
}