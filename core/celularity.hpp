#pragma once
#include "types.hpp"
#include "terminality.hpp"

namespace innate {
	enum cell_type {
		cell_simple,
		cell_exre
	};

	struct cell {
		cell_type type;
		int width = -1;
		int height = -1;
		int spillover = 0;
	};
;
	struct cell_simple : public cell {
		int threshold_activation = 2;
	};

	struct cell_exre : public cell {
		int tacts_excitation = 1;
		int tacts_relaxation = 1;
	};
}

namespace data {
	enum cell_expression {
		alive      = 0x00000001,
		depression = 0x00000010
	};
	struct __align_4b__ cell {
		state8_t expression;
		rgstr8_t spikes;
	};

	enum cell_simple_expression {
	};
	struct __align_4b__ cell_simple : public cell {

	};

	enum cell_exre_expression {
	};
	struct __align_4b__ cell_exre : public cell {
		uint8_t counter_tacts_excitation;
		uint8_t counter_tacts_relaxation;
	};

	struct celular {
		__const__ innate::cell* innate = nullptr; // cast for innate::cell_type

		__mem__ cell* cells = nullptr;            // cast for innate::cell_type, memory alocate array
		__mem__ float* results = nullptr;         // size = cluster_count * cell::width * cell::height * 4
	};
}
