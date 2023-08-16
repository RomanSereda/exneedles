#pragma once
#pragma warning(disable : 26812)
#pragma warning(disable : 26495)
#pragma warning(disable : 26451)
#pragma warning(disable : 26819)
#pragma warning(disable : 26800)
#pragma warning(disable : 6294)
#pragma warning(disable : 6201)

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
		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i == sz_args), void>::type foreach(T0* t0, F f) { }
		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i < sz_args), void>::type foreach(T0* t0, F f) {
			arg<i> sample;
			if (sample.type == t0->type) {
				f(static_cast<arg<i>*>(t0));
			}
			foreach<i + 1, T0, F>(t0, f);
		}

		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i == sz_args), void>::type to(T0* t0, F f) { }
		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i < sz_args), void>::type to(T0* t0, F f) {
			arg<i> sample;
			if (sample.type == t0->type) {
				return f(static_cast<arg<i>*>(t0));
			}
			return to<i + 1, T0, F>(t0, f);
		}
	
		template<std::size_t i = 0, typename T, typename F> static typename std::enable_if<(i == sz_args), void>::type create(T type, F f) {}
		template<std::size_t i = 0, typename T, typename F> static typename std::enable_if<(i < sz_args), void>::type create(T type, F f) {
			using Type = arg<i>;
			std::unique_ptr<Type> sample(new Type());
			if (sample->type == type) {
				f(std::move(sample));
				return;
			}
			return create<i + 1, T, F>(type, f);
		}

		template<std::size_t i = 0, typename T0> static typename std::enable_if<(i == sz_args), int>::type size(T0* t0) { return 0; }
		template<std::size_t i = 0, typename T0> static typename std::enable_if<(i < sz_args), int>::type size(T0* t0) {
			arg<i> sample;
			if (sample.type == t0->type) 
				return sizeof(sample);
			
			return size<i + 1, T0>(t0);
		}
	};

	template<typename... args> class spec_pair_tuple {
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
			}
			foreach<i + 1, T0, T1, F>(t0, t1, f);
		}

		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i == sz_args), void>::type foreach(T0* t0, F f) { }
		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i < sz_args), void>::type foreach(T0* t0, F f) {
			arg<i> sample;
			if ((std::get<0>(sample)).type == t0->type) {
				using Type0 = typename std::tuple_element<0, arg<i>>::type;
				f(static_cast<Type0*>(t0));
			}
			foreach<i + 1, T0, F>(t0, f);
		}

		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i == sz_args), void>::type to_first(T0* t0, F f) { }
		template<std::size_t i = 0, typename T0, typename F> static typename std::enable_if<(i < sz_args), void>::type to_first(T0* t0, F f) {
			arg<i> sample;
			if ((std::get<0>(sample)).type == t0->type) {
				using Type0 = typename std::tuple_element<0, arg<i>>::type;
				f(static_cast<Type0*>(t0));
				return;
			}
			return to_first<i + 1, T0, F>(t0, f);
		}

		template<std::size_t i = 0, typename T0> static typename std::enable_if<(i == sz_args), std::vector<size_t>>::type size(T0* t0) { return {}; }
		template<std::size_t i = 0, typename T0> static typename std::enable_if<(i < sz_args), std::vector<size_t>>::type size(T0* t0) {
			arg<i> sample;
			if ((std::get<0>(sample)).type == t0->type) {
				return { sizeof(typename std::tuple_element<0, arg<i>>::type), 
					     sizeof(typename std::tuple_element<1, arg<i>>::type) 
				};
			}
			return size<i + 1, T0>(t0);
		}
	
	
		template<std::size_t i = 0, typename T, typename F> static typename std::enable_if<(i == sz_args), void>::type create_first(T type, F f) {}
		template<std::size_t i = 0, typename T, typename F> static typename std::enable_if<(i < sz_args), void>::type create_first(T type, F f) {
			arg<i> sample;
			if ((std::get<0>(sample)).type == type) {
				using Type0 = typename std::tuple_element<0, arg<i>>::type;
				f(std::move(std::unique_ptr<Type0>(new Type0())));
				return;
			}
			return create_first<i + 1, T, F>(type, f);
		}
	};

	static std::string to_string(ptree tree) {
		std::stringstream s;
		write_json(s, tree);
		return s.str();
	}

	template<typename T> auto to_ptree(T var) {
		ptree root;
		hana::for_each(var, hana::fuse([&](auto member, auto value) {
			root.put(hana::to<char const*>(member), std::to_string(value));
		}));
		return root;
	}

	template <typename T> std::enable_if_t<hana::Struct<T>::value, T> to(T& var, ptree root) {
		hana::for_each(hana::keys(var), [&](auto key) {
			auto& value = hana::at_key(var, key);
			using member = std::remove_reference_t<decltype(value)>;
			auto it = root.find(hana::to<char const*>(key));
			value = static_cast<member>(std::stoi(it->second.data()));
		});
		return var;
	}
}

#pragma warning(default : 26812)
#pragma warning(default : 26495)
#pragma warning(default : 26451)
#pragma warning(default : 26819)
#pragma warning(default : 26800)
#pragma warning(default : 6294)
#pragma warning(default : 6201)