#pragma once
#pragma warning(disable : 26812)
#include <tuple>
#include <utility> 
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/hana/adapt_struct.hpp>
#include <boost/hana/take_front.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/at_key.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/fuse.hpp>
#include <boost/hana/keys.hpp>
#include <boost/hana/at.hpp>
#include <boost/json.hpp>

#include "types.hpp"

#pragma comment(lib, "libboost_json-vc143-mt-gd-x64-1_82.lib")

namespace hana = boost::hana;
using ptree = boost::property_tree::ptree;


namespace boost {
	using namespace boost::property_tree;

	template<typename... args> class spec_tuple {
	protected:
		static const int sz_args = sizeof...(args);
		template <int d> using arg = typename std::tuple_element<d, std::tuple<args...>>::type;

	public:
		template<std::size_t i = 0, typename P, typename F> static std::enable_if<(i == sz_args), void>::type foreach(P* p, F f) { }
		template<std::size_t i = 0, typename P, typename F> static std::enable_if<(i < sz_args), void>::type foreach(P* p, F f) {
			if (!p) return;

			arg<i> m;
			if(p->type == m.type)
				if(auto pp = dynamic_cast<arg<i>>(p))
					if (f(pp)) return;

			foreach<i + 1, P, F>(p, f);
		}
	};
}
