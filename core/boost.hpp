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
