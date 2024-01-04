#pragma once
#include "types.hpp"
#include "boost.hpp"

#define LIBRARY "core"
#include "../deflib.inc"

namespace innate {
	struct size {
		int width;
		int height;
	};
}

BOOST_HANA_ADAPT_STRUCT(innate::size, width, height);

namespace instance {
	struct readable_splvr_innate;
	struct readable_cell_innate;
	struct readable_splvr_instance;
	struct readable_cell_instance;

	class icellularity_host_accessor;
	class ispilloverity_host_accessor;

	struct readable_layer_innate {
		const std::vector<readable_splvr_innate> spillovers;
		const std::vector<readable_cell_innate> cellularity;
	};
	struct readable_layer_instance {
		const std::vector<readable_splvr_instance> spillovers;
		const std::vector<readable_cell_instance> cellularity;
	};

	class LIBRARY_API ilayerality {
	public:
		virtual const innate::size& size() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_layer_innate innate() const = 0;
		virtual readable_layer_instance instance() const = 0;
	};

	class LIBRARY_API ilayerality_host_accessor {
	public:
		virtual ilayerality& layerality() = 0;

		virtual void rm_cell(const std::string& id) = 0;
		virtual void rm_splvr(const std::string& id) = 0;

		virtual icellularity_host_accessor& add_cell(const std::string& id) = 0;
		virtual ispilloverity_host_accessor& add_splvr(const std::string& id) = 0;

		virtual void get_cells(std::unordered_map<std::string, icellularity_host_accessor&>& cells) const = 0;
		virtual void get_splvrs(std::unordered_map<std::string, ispilloverity_host_accessor&>& splvrs) const = 0;
	};

	struct readable_region_innate {
		const innate::size size;
		const std::vector<readable_layer_innate> layers;
	};
	struct readable_region_instance {
		const std::vector<readable_layer_instance> layers;
	};

	class LIBRARY_API iregion {
	public:
		virtual const innate::size& size() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_region_innate innate() const = 0;
		virtual readable_region_instance instance() const = 0;
	};

	class LIBRARY_API iregion_host_accessor {
	public:
		virtual iregion& region() = 0;

		virtual void rm_layer(const std::string& id) = 0;
		virtual ilayerality_host_accessor& add_layer(const std::string& id) = 0;
		virtual void get_layers(std::unordered_map<std::string, ilayerality_host_accessor&>& layers) const = 0;
	};
}