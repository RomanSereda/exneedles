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

	class layerality_host : public layerality_cpu_type, public ilayerality_host_accessor {
	public:
		layerality_host(const ptree& root, const innate::size& size);

	public:
		ilayerality& layerality() override;

		void rm_cell(const std::string& id) override;
		void rm_splvr(const std::string& id) override;

		icellularity_host_accessor& add_cell(const std::string& id, innate::cell::cell_type deftype) override;
		ispilloverity_host_accessor& add_splvr(const std::string& id, innate::spillover::spillover_type deftype) override;

		void get_cells(std::unordered_map<std::string, icellularity_host_accessor&>& cells) const override;
		void get_splvrs(std::unordered_map<std::string, ispilloverity_host_accessor&>& splvrs) const override;

	private:
		std::unordered_map<std::string, cellularity_cpu_type*> m_icellularitys;
		std::unordered_map<std::string, spilloverity_cpu_type*> m_ispilloveritys;

		icellularity_host_accessor& add_cell(const std::string& id, const ptree& root, const innate::size& size, 
			                                 innate::cell::cell_type deftype = innate::cell::cell_simple);

		ispilloverity_host_accessor& add_splvr(const std::string& id, const ptree& root, const innate::size& size, 
			                                   innate::spillover::spillover_type deftype = innate::spillover::simple_spillover);
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

	class region_host : public region_cpu_type, public iregion_host_accessor {
	public:
		region_host(const ptree& root);

	public:
		iregion& region() override;

		void rm_layer(const std::string& id) override;
		ilayerality_host_accessor& add_layer(const std::string& id) override;
		void get_layers(std::unordered_map<std::string, ilayerality_host_accessor&>& layers) const override;

	private:
		ilayerality_host_accessor& add_layer(const std::string& id, const ptree& root, const innate::size& size);
		std::unordered_map<std::string, layerality_cpu_type*> m_ilayeralitys;
	};

	class region_device : public region_gpu_type {
	public:
		region_device(const ptree& root);
	};
}