#pragma once
#include "types.hpp"
#include "boost.hpp"

namespace innate {
	struct terminal {
		enum terminal_type {
			axon_simple = 0,
			synapse_simple
		} type;
		terminal(terminal_type t = axon_simple) : type{ t } {};
	};


	struct axon_simple: public terminal {
		axon_simple() : terminal(terminal_type::axon_simple ){};
		int basic_value = 1;
	};

	struct synapse_simple: public terminal {
		enum terminal_sign {
			positive = 0,
			negative
		} sign;
		synapse_simple() : terminal(terminal_type::synapse_simple), sign{ positive } {};
	};

	struct cluster {
		enum cluster_type {
			cluster_targeted
		} type;

		cluster(cluster_type t = cluster_targeted) : type{ t } {};

		int width = -1;
		int height = -1;
	};

	struct cluster_targeted: public cluster {
		cluster_targeted() : cluster(cluster_type::cluster_targeted) {};

		int target_layer = -1;
		int target_region = -1;
	};
}

namespace data {
	struct __align_4b__ terminal {
		enum terminal_expression {
			alive = 0x00000001
		};

		state8_t expression;
		rgstr8_t spikes;
	};

	struct __align_4b__ axon_simple : terminal {
		enum axon_simple_expression {
			depression = 0x00000010
		};

	};

	struct __align_4b__ synapse_simple : terminal {
		enum synapse_simple_expression {
			augumentation = 0x00000010
		};

	};
}

namespace innate { struct layer; }
namespace instance {
	using cluster_tuple = boost::spec_tuple<innate::cluster_targeted>;
	using cluster_data_tuple = boost::spec_pair_tuple<std::tuple<innate::axon_simple,    data::axon_simple>,
		                                              std::tuple<innate::synapse_simple, data::synapse_simple>>;

	class terminality {
	protected:
		std::tuple<
			__const__ innate::cluster*,
			__const__ innate::terminal*> innate{ nullptr, nullptr };

		__mem__ float* results = nullptr;                       // bytes = layer::celulars_count * cell::width * cell::height * 4 --> shift celular number
		__mem__ data::terminal* terminals = nullptr;            // cast for terminal_type, memory alocate array.  
		                                                        // bytes = cell::width * cell::height * cluster::width * cluster::height * sizeof(terminal_type) --> shift cluster number
	
		virtual void* malloc(int size) const = 0;
	};

	class host_terminality : public terminality {
	public:
		host_terminality(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;

	protected:
		void* malloc(int size) const override;

	};

	class device_terminality: public terminality {
	protected:
		void* malloc(int size) const override;
	};
	
}

BOOST_HANA_ADAPT_STRUCT(innate::terminal, type);
BOOST_HANA_ADAPT_STRUCT(innate::axon_simple, basic_value);
BOOST_HANA_ADAPT_STRUCT(innate::synapse_simple, sign);

BOOST_HANA_ADAPT_STRUCT(innate::cluster, type, width, height);
BOOST_HANA_ADAPT_STRUCT(innate::cluster_targeted, target_layer, target_region);