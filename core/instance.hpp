#pragma once
#include "terminality.hpp"
#include "memory.cuh"

namespace instance {
	class LIBRARY_API host_terminality : public terminality<UPTR_TEMPLATE> {
	public:
		host_terminality(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;
	};

	class LIBRARY_API device_terminality : public terminality<PTR_TEMPLATE> {
	public:
		device_terminality(const ptree& root, const innate::layer& layer);
		virtual ~device_terminality();

		memory::const_empl::ptr const_emplace_cl() const;
		memory::const_empl::ptr const_emplace_tr() const;

	private:
		memory::const_empl::ptr m_const_cl = nullptr;
		memory::const_empl::ptr m_const_tr = nullptr;

		void setup_const_memory(const std::unique_ptr<innate::cluster>& cl, 
			                    const std::unique_ptr<innate::terminal>& tr);
	};

}
