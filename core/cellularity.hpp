#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "../deflib.inc"

namespace innate {
	struct LIBRARY_API cell {
		enum cell_type {
			cell_simple,
			cell_exre
		} type;

		cell(cell_type t = cell_simple);
	};
;
	struct LIBRARY_API cell_simple : public cell {
		cell_simple();
	};

	struct LIBRARY_API cell_exre : public cell {
		cell_exre();

		int tacts_excitation = 1;
		int tacts_relaxation = 1;
	};
}

BOOST_HANA_ADAPT_STRUCT(innate::cell, type);
BOOST_HANA_ADAPT_STRUCT(innate::cell_simple);
BOOST_HANA_ADAPT_STRUCT(innate::cell_exre, tacts_excitation, tacts_relaxation);

namespace data {
	struct __align_4b__ cell {
		enum cell_expression {
			alive = 0x00000001,
			depression = 0x00000010
		};

		state8_t expression;
		rgstr8_t spikes;
	};

	struct __align_4b__ cell_simple : public cell {
		enum cell_simple_expression {
		};

	};

	struct __align_4b__ cell_exre : public cell {
		enum cell_exre_expression {
		};

		uint8_t counter_tacts_excitation;
		uint8_t counter_tacts_relaxation;
	};
}

using cell_data_tuple = boost::spec_pair_tuple<std::tuple<innate::cell_simple, data::cell_simple>,
	                                           std::tuple<innate::cell_exre, data::cell_exre>>;

namespace innate { struct size; }
namespace instance {
	struct readable_trmn_innate;
	struct readable_trmn_instance;

	struct readable_cell_innate { 
		const innate::cell* cell; 
		const std::vector<readable_trmn_innate> terminality;
	};
	struct readable_cell_instance {
		__mem__ void*  cells   = nullptr;
		__mem__ float* results = nullptr;
		size_t cells_szb   = -1;
		size_t results_szb = -1;

		const std::vector<readable_trmn_instance> terminality;
	};

	class LIBRARY_API icellularity {
	public:
		virtual const innate::size& size() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_cell_innate innate() const = 0;
		virtual readable_cell_instance instance() const = 0;

		static std::unique_ptr<innate::cell> to_innate(const ptree& root);
		static ptree to_ptree(innate::cell* c);

	protected:
		virtual __mem__ float* results() const = 0;
		virtual __mem__ void* cells() const = 0;

		virtual size_t results_szb() const = 0;
		virtual size_t cells_szb() const = 0;

		static size_t calc_results_bytes(const innate::size& size);
		static size_t calc_cells_bytes(const innate::size& size, const innate::cell* c);
	};
}