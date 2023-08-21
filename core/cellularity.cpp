#include "cellularity.hpp"

namespace innate {
	cell::cell(cell_type t): type(t){}
	cell_simple::cell_simple() : cell(cell_type::cell_simple) {};
	cell_exre::cell_exre() : cell(cell_type::cell_exre) {};
}

namespace instance {
	template<typename T, typename TR> 
	const T& celularity<T, TR>::inncell() const {
		return m_innate;
	}

	template<typename T, typename TR> 
	const innate::layer& celularity<T, TR>::layer() const {
		return m_layer;
	}

	template<typename T, typename TR> 
	celularity<T, TR>::celularity(const innate::layer& layer) : m_layer(layer) {}
	template<typename T, typename TR> 
	celularity<T, TR>::~celularity(){};


}