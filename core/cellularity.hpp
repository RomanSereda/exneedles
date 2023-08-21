#pragma once
#include "types.hpp"
#include "boost.hpp"

#include "terminality.hpp"

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

namespace instance {
	using cell_data_tuple = boost::spec_pair_tuple<std::tuple<innate::cell_simple, data::cell_simple>,
		                                           std::tuple<innate::cell_exre,   data::cell_exre>>;

	template<typename T, typename TR> class LIBRARY_API cellularity {
	public:
		const T& inncell() const;
		const innate::layer& layer() const;

		static std::unique_ptr<innate::cell> to_innate(const ptree& root);
		static ptree to_ptree(innate::cell* c);

		virtual ~cellularity();

	protected:
		cellularity(const innate::layer& layer);

		size_t calc_results_bytes(const innate::layer& layer) const;
		size_t calc_cells_bytes(const innate::layer& layer, const innate::cell* c) const;

		T m_innate = nullptr;                             // cast for innate::cell_type

		__mem__ data::cell* m_cells = nullptr;            // cast for innate::cell_type, memory alocate array
		__mem__ float* m_results = nullptr;               // size = cluster_count * cell::width * cell::height * 4

		size_t m_cells_szb = 0;
		size_t m_results_szb = 0;

		const innate::layer& m_layer;
	};


#define PTR_TEMPLATE_CELL  cellularity<__const__ innate::cell**, instance::terminality<PTR_TEMPLATE_TR>>
#define UPTR_TEMPLATE_CELL cellularity<std::unique_ptr<innate::cell>, instance::terminality<UPTR_TEMPLATE_TR>>

	EXPIMP_TEMPLATE template class LIBRARY_API UPTR_TEMPLATE_CELL;
	EXPIMP_TEMPLATE template class LIBRARY_API PTR_TEMPLATE_CELL;

}

BOOST_HANA_ADAPT_STRUCT(innate::cell, type);
BOOST_HANA_ADAPT_STRUCT(innate::cell_simple);
BOOST_HANA_ADAPT_STRUCT(innate::cell_exre, tacts_excitation, tacts_relaxation);
