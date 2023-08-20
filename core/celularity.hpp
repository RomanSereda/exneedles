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

		int width = -1;
		int height = -1;
		int spillover = 0;
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
	template<typename T, typename TR> class LIBRARY_API celularity {
	public:
		const T& inncell() const;
		const innate::layer& layer() const;

		virtual const std::vector<std::unique_ptr<TR>>& cltr_instances() const = 0;

		virtual ~celularity();

	protected:
		celularity(const innate::layer& layer);

	private:
		T m_innate = nullptr;                             // cast for innate::cell_type

		__mem__ data::cell* m_cells = nullptr;            // cast for innate::cell_type, memory alocate array
		__mem__ float* m_results = nullptr;               // size = cluster_count * cell::width * cell::height * 4

		size_t m_cells_szb = 0;
		size_t m_results_szb = 0;

		const innate::layer& m_layer;
	};

#define PTR_TEMPLATE_CELL      __const__ innate::cell**
#define UPTR_TEMPLATE_CELL std::unique_ptr<innate::cell>

#define UPTR_TEMPLATE_CELULARITY celularity<UPTR_TEMPLATE_CELL, terminality<UPTR_TEMPLATE_TR>>
#define PTR_TEMPLATE_CELULARITY  celularity<PTR_TEMPLATE_CELL, terminality<PTR_TEMPLATE_TR>>

	EXPIMP_TEMPLATE template class LIBRARY_API UPTR_TEMPLATE_CELULARITY;
	EXPIMP_TEMPLATE template class LIBRARY_API PTR_TEMPLATE_CELULARITY;

}
