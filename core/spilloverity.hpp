#pragma once
#include "types.hpp"
#include "boost.hpp"

#define LIBRARY "core"
#include "../deflib.inc"

namespace innate {
	struct LIBRARY_API spillover {
		enum spillover_type {
			simple_spillover
		} type;

		spillover(spillover_type t = simple_spillover);
	};

	LIBRARY_API void get_items(std::vector<spillover::spillover_type>& items);
	LIBRARY_API std::string to_string(spillover::spillover_type type);

	struct LIBRARY_API simple_spillover : public spillover {
		simple_spillover();
	};
}

namespace data {
	struct __align_4b__ spillover {
	};

	struct __align_4b__ simple_spillover : public spillover {
	};
}

BOOST_HANA_ADAPT_STRUCT(innate::spillover, type);
BOOST_HANA_ADAPT_STRUCT(innate::simple_spillover);

using spillover_data_tuple = boost::spec_pair_tuple<std::tuple<innate::simple_spillover, data::simple_spillover>>;

namespace innate { struct size; }
namespace instance {
	struct readable_splvr_innate {
		const innate::spillover* splvr;
	};
	struct readable_splvr_instance {
		__mem__ void* spillovers = nullptr;
		size_t spillovers_szb = - 1;
	};

	class LIBRARY_API ispilloverity {
	public:
		virtual const innate::size& size() const = 0;
		virtual ptree to_ptree() const = 0;

		virtual readable_splvr_innate innate() const = 0;
		virtual readable_splvr_instance instance() const = 0;

		static std::unique_ptr<innate::spillover> to_innate(const ptree& root, 
			innate::spillover::spillover_type deftype = innate::spillover::simple_spillover);

		static ptree to_ptree(innate::spillover* c);

	protected:
		virtual __mem__ void* spillovers() const = 0;
		virtual size_t spillovers_szb() const = 0;

		static size_t calc_spillovers_bytes(const innate::size& size,
			                                const innate::spillover* splvr);
	};

	class LIBRARY_API ispilloverity_host_accessor {
	public:
		virtual ispilloverity& spilloverity() = 0;
	};

}
