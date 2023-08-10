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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "types.hpp"

#pragma comment(lib, "libboost_json-vc143-mt-gd-x64-1_82.lib")

namespace hana = boost::hana;
using ptree = boost::property_tree::ptree;

namespace boost {

	template<typename... args> class spec_tuple {
	protected:
		static const int sz_args = sizeof...(args);
		template <int d> using arg = typename std::tuple_element<d, std::tuple<args...>>::type;

	public:
		template<std::size_t i = 0, typename T0, typename T1, typename F> static typename std::enable_if<(i == sz_args), void>::type foreach(T0* t0, T1* t1, F f) { }
		template<std::size_t i = 0, typename T0, typename T1, typename F> static typename std::enable_if<(i < sz_args), void>::type foreach(T0* t0, T1* t1, F f) {
			arg<i> sample;
			if ((std::get<0>(sample)).type == t0->type) {
				using Type0 = typename std::tuple_element<0, arg<i>>::type;
				using Type1 = typename std::tuple_element<1, arg<i>>::type;

				auto p0 = static_cast<Type0*>(t0);
				auto p1 = static_cast<Type1*>(t1);

				f(p0, p1);

				return;
			}
			
			foreach<i + 1, T0, T1, F>(t0, t1, f);
		}

		template<std::size_t i = 0, typename T0> static typename std::enable_if<(i == sz_args), std::vector<size_t>>::type size_of_types(T0* t0) { return {}; }
		template<std::size_t i = 0, typename T0> static typename std::enable_if<(i < sz_args), std::vector<size_t>>::type size_of_types(T0* t0) {
			arg<i> sample;
			if ((std::get<0>(sample)).type == t0->type) {
				return { sizeof(typename std::tuple_element<0, arg<i>>::type), 
					     sizeof(typename std::tuple_element<1, arg<i>>::type) 
				};
			}
			return size_of_types<i + 1, T0>(t0);
		}
	};

	template<typename T> auto to_ptree(T var) {
		ptree root;
		/*hana::for_each(var, hana::fuse([&](auto member, auto value) {
			root.put(hana::to<char const*>(member), std::to_string(value));
			}));*/
		return root;
	}

	template <typename T> std::enable_if_t<hana::Struct<T>::value, T> to(T& var, ptree root) {
		/*hana::for_each(hana::keys(var), [&](auto key) {
			auto& value = hana::at_key(var, key);
			using member = std::remove_reference_t<decltype(value)>;
			auto it = root.find(hana::to<char const*>(key));
			value = static_cast<member>(std::stoi(it->second.data()));
			});*/
		return var;
	}
}
