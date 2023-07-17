#pragma once
#include "types.hpp"

namespace innate {
	enum terminal_type {
		axon_simple,
		synapse_simple
	};

	enum terminal_sign {
		positive = 0,
		negative
	};

	struct terminal {
		terminal_type type;
		terminal_sign sign;
		int basic_value = 1;
	};

	struct cluster {
		cluster_type type;
		int width = -1;
		int height = -1;
		int target_layer = -1;
		int target_region = -1;
	};

	enum cluster_type {
		cluster_simple
	};

	struct cluster_simple : public cluster, public terminal {};
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
		__mem__ float* results = nullptr;                 // bytes = layer::celulars_count * cell::width * cell::height * 4 --> shift celular number
		__mem__ terminal* terminals = nullptr;            // cast for terminal_type, memory alocate array.  
														  // bytes = cell::width * cell::height * cluster::width * cluster::height * sizeof(terminal_type) --> shift cluster number
	};
}