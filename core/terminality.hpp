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

BOOST_HANA_ADAPT_STRUCT(innate::terminal, type);
BOOST_HANA_ADAPT_STRUCT(innate::axon_simple, basic_value);
BOOST_HANA_ADAPT_STRUCT(innate::synapse_simple, sign);

BOOST_HANA_ADAPT_STRUCT(innate::cluster, type, width, height);
BOOST_HANA_ADAPT_STRUCT(innate::cluster_targeted, target_layer, target_region);

using cluster_tuple = boost::spec_tuple<innate::cluster_targeted>;
using cluster_data_tuple = boost::spec_pair_tuple<std::tuple<innate::axon_simple, data::axon_simple>,
	                                              std::tuple<innate::synapse_simple, data::synapse_simple>>;


#define PTR_TEMPLATE_TR       __const__ innate::cluster**,      __const__ innate::terminal**
#define UPTR_TEMPLATE_TR std::unique_ptr<innate::cluster>, std::unique_ptr<innate::terminal>

struct readable_cltr_innate { 
	const innate::cluster* cl; 
	const innate::terminal* tr; 
};

namespace innate { struct layer; }

namespace instance {
	class LIBRARY_API iterminality {
	public:
		virtual const innate::layer& layer() const = 0;
		virtual ptree to_ptree() const = 0;
		virtual readable_cltr_innate innate() const = 0;

		virtual __mem__ float* results() const = 0;
		virtual __mem__ void* terminals() const = 0;

		virtual size_t results_szb() const = 0;
		virtual size_t terminals_szb() const = 0;

		static std::tuple<UPTR_TEMPLATE_TR> to_innate(const ptree& root);
		static ptree to_ptree(innate::cluster* cl, innate::terminal* tr);

	protected:
		static size_t calc_results_bytes(const innate::layer& layer);
		static size_t calc_terminals_bytes(const innate::layer& layer,
			                               const innate::cluster* cl,
			                               const innate::terminal* tr);
	private:
		static std::unique_ptr<innate::cluster> to_inncl(const ptree& root);
		static std::unique_ptr<innate::terminal> to_inntr(const ptree& root);

		static ptree to_ptree(innate::cluster* cl);
		static ptree to_ptree(innate::terminal* tr);
	};
}