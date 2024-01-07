#pragma once
#include "types.hpp"
#include "boost.hpp"

#define LIBRARY "core"
#include "../deflib.inc"

namespace innate {
	struct LIBRARY_API terminal {
		enum terminal_sign {
			positive = 0,
			negative
		} sign;

		enum terminal_type {
			axon_simple = 0,
			synapse_simple
		} type;

		terminal(terminal_type t = axon_simple);
	};


	LIBRARY_API void get_items(std::vector<terminal::terminal_type>& items);
	LIBRARY_API std::string to_string(terminal::terminal_type type);

	LIBRARY_API void get_items(std::vector<terminal::terminal_sign>& items);
	LIBRARY_API std::string to_string(terminal::terminal_sign type);

	struct LIBRARY_API axon_simple: public terminal {
		axon_simple();
		int basic_value = 1;
	};

	struct LIBRARY_API synapse_simple: public terminal {
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

	LIBRARY_API void get_items(std::vector<cluster::cluster_type>& items);
	LIBRARY_API std::string to_string(cluster::cluster_type type);

	struct LIBRARY_API cluster_targeted: public cluster {
		cluster_targeted();

		int target_layer_index = -1;
		int target_spillover_index = -1;
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

BOOST_HANA_ADAPT_STRUCT(innate::terminal, sign, type);
BOOST_HANA_ADAPT_STRUCT(innate::axon_simple, basic_value);
BOOST_HANA_ADAPT_STRUCT(innate::synapse_simple);

BOOST_HANA_ADAPT_STRUCT(innate::cluster, type, width, height);
BOOST_HANA_ADAPT_STRUCT(innate::cluster_targeted, target_layer_index, target_spillover_index);

using cluster_tuple = boost::spec_tuple<innate::cluster_targeted>;
using cluster_data_tuple = boost::spec_pair_tuple<std::tuple<innate::axon_simple, data::axon_simple>,
	                                              std::tuple<innate::synapse_simple, data::synapse_simple>>;


#define PTR_TEMPLATE_TR       __const__ innate::cluster**,      __const__ innate::terminal**
#define UPTR_TEMPLATE_TR std::unique_ptr<innate::cluster>, std::unique_ptr<innate::terminal>


namespace innate { struct size; }
namespace instance {
	struct readable_trmn_innate {
		const innate::cluster* cl;
		const innate::terminal* tr;
	};
	struct readable_trmn_instance {
		__mem__ void* terminals = nullptr;
		__mem__ float* results = nullptr;
		size_t terminals_szb = -1;
		size_t results_szb = -1;
	};

	class LIBRARY_API iterminality {
	public:
		virtual const innate::size& size() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_trmn_innate innate() const = 0;
		virtual readable_trmn_instance instance() const = 0;

		struct InnateTerminalityParam {
			innate::terminal::terminal_type tr_type 
				= innate::terminal::synapse_simple;

			innate::cluster::cluster_type cl_type 
				= innate::cluster::cluster_targeted;

			int width = -1;
			int height = -1;
		};
		
		static std::tuple<UPTR_TEMPLATE_TR> to_innate(
			const ptree& root, const InnateTerminalityParam& def = InnateTerminalityParam());

		static ptree to_ptree(innate::cluster* cl, innate::terminal* tr);

	protected:
		virtual __mem__ float* results() const = 0;
		virtual __mem__ void* terminals() const = 0;

		virtual size_t results_szb() const = 0;
		virtual size_t terminals_szb() const = 0;

		static size_t calc_results_bytes(const innate::size& size);
		static size_t calc_terminals_bytes(const innate::size& size,
			                               const innate::cluster* cl,
			                               const innate::terminal* tr);
	private:
		static std::unique_ptr<innate::cluster> to_inncl(const ptree& root, 
			innate::cluster::cluster_type deftype = innate::cluster::cluster_targeted, int width = -1, int height = -1);

		static std::unique_ptr<innate::terminal> to_inntr(const ptree& root, 
			innate::terminal::terminal_type deftype = innate::terminal::synapse_simple);

		static ptree to_ptree(innate::cluster* cl);
		static ptree to_ptree(innate::terminal* tr);
	};

	class LIBRARY_API iterminality_host_accessor {
	public:
		virtual iterminality& terminality() = 0;
	};
}