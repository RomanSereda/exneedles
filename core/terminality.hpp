#pragma once
#include "types.hpp"
#include "celularity.hpp"
#include "boost.hpp"

namespace innate {

	enum terminal_sign {
		positive = 0,
		negative
	};

	enum terminal_type {
		axon_simple,
		synapse_simple
	};
	struct terminal {
		terminal_type type;
	};

	struct terminal_axon_simple: public terminal {
		int basic_value = 1;
	};

	struct terminal_synapse_simple: public terminal {
		terminal_sign sign;
	};


	enum cluster_type {
		cluster_targeted
	};
	struct cluster {
		cluster_type type;
		int width = -1;
		int height = -1;
	};

	struct cluster_targeted: public cluster {
		int target_layer = -1;
		int target_region = -1;
	};
}

namespace data {
	enum terminal_expression {
		alive = 0x00000001
	};
	struct __align_4b__ terminal {
		state8_t expression;
		rgstr8_t spikes;
	};

	enum terminal_axon_simple_expression {
		depression = 0x00000010
	};
	struct __align_4b__ terminal_axon_simple : terminal {
	};

	enum terminal_synapse_simple_expression {
		augumentation = 0x00000010
	};
	struct __align_4b__ terminal_synapse_simple : terminal {
	};

	struct cluster {
		std::tuple<
			__const__ innate::cluster*, 
			__const__ innate::terminal*> innate {nullptr, nullptr};

		__mem__ float* results = nullptr;                       // bytes = layer::celulars_count * cell::width * cell::height * 4 --> shift celular number
		__mem__ terminal* terminals = nullptr;                  // cast for terminal_type, memory alocate array.  
														        // bytes = cell::width * cell::height * cluster::width * cluster::height * sizeof(terminal_type) --> shift cluster number
	};
}

namespace instance {
	class cluster {
	public:
		cluster& operator=(const ptree& root)
		{

			return *this;
		}
	};
}