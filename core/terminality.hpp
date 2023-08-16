#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "../deflib.inc"

namespace innate {
	struct LIBRARY_API terminal {
		enum terminal_type {
			axon_simple = 0,
			synapse_simple
		} type;
		terminal(terminal_type t = axon_simple);
	};

	struct LIBRARY_API axon_simple: public terminal {
		axon_simple();
		int basic_value = 1;
	};

	struct LIBRARY_API synapse_simple: public terminal {
		enum terminal_sign {
			positive = 0,
			negative
		} sign;
		synapse_simple();
	};

	struct LIBRARY_API cluster {
		enum cluster_type {
			cluster_targeted
		} type;

		cluster(cluster_type t = cluster_targeted);

		int width = -1;
		int height = -1;
	};

	struct LIBRARY_API cluster_targeted: public cluster {
		cluster_targeted();

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
	using cluster_tuple      = boost::spec_tuple<innate::cluster_targeted>;
	using cluster_data_tuple = boost::spec_pair_tuple<std::tuple<innate::axon_simple,    data::axon_simple>,
		                                              std::tuple<innate::synapse_simple, data::synapse_simple>>;

	template<typename T0, typename T1> class LIBRARY_API terminality {
	public:
		std::tuple<T0, T1> innate{ nullptr, nullptr };

		__mem__ float* results = nullptr;                       // bytes = layer::celulars_count * cell::width * cell::height * 4 --> shift celular number
		__mem__ data::terminal* terminals = nullptr;            // cast for terminal_type, memory alocate array.  
		                                                        // bytes = cell::width * cell::height * cluster::width * cluster::height * sizeof(terminal_type) --> shift cluster number
	
		const T0& inncl() const;
		const T1& inntr() const;

		static std::unique_ptr<innate::cluster>&& toinncl(const ptree& root);
		static std::unique_ptr<innate::terminal>&& toinntr(const ptree& root);
		
		size_t calc_results_bytes(const innate::layer& layer) const;
		size_t calc_terminals_bytes(const innate::layer& layer,
			                        const innate::cluster* cl,
			                        const innate::terminal* tr) const;

	};

	class LIBRARY_API host_terminality : protected terminality<std::unique_ptr<innate::cluster>,
		                                                       std::unique_ptr<innate::terminal>> {
	public:
		host_terminality(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;
	};

	class LIBRARY_API device_terminality: protected terminality<__const__ innate::cluster*,
		                                                        __const__ innate::terminal*> {
	public:
		device_terminality(const ptree& root, const innate::layer& layer);
		~device_terminality();
	};
	
}

BOOST_HANA_ADAPT_STRUCT(innate::terminal, type);
BOOST_HANA_ADAPT_STRUCT(innate::axon_simple, basic_value);
BOOST_HANA_ADAPT_STRUCT(innate::synapse_simple, sign);

BOOST_HANA_ADAPT_STRUCT(innate::cluster, type, width, height);
BOOST_HANA_ADAPT_STRUCT(innate::cluster_targeted, target_layer, target_region);