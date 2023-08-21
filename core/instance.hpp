#pragma once
#include "terminality.hpp"
#include "cellularity.hpp"
#include "memory.cuh"

namespace instance {
	class LIBRARY_API host_terminality : public terminality<UPTR_TEMPLATE_TR> {
	public:
		host_terminality(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;
	};

	class LIBRARY_API device_terminality : public terminality<PTR_TEMPLATE_TR> {
	public:
		device_terminality(const ptree& root, const innate::layer& layer);
		virtual ~device_terminality();

		memory::const_empl::ptr const_emplace_cl() const;
		memory::const_empl::ptr const_emplace_tr() const;

	private:
		memory::const_empl::ptr m_const_cl = nullptr;
		memory::const_empl::ptr m_const_tr = nullptr;

		void setup_const_memory(innate::cluster* cl, innate::terminal* tr);
	};
}

namespace instance {
	class LIBRARY_API host_celularity : public UPTR_TEMPLATE_CELL {
	public:
		host_celularity(const ptree& root, const innate::layer& layer);
		ptree to_ptree() const;

	private:
	
	};

	class LIBRARY_API device_celularity : public PTR_TEMPLATE_CELL {
	public:
		device_celularity(const ptree& root, const innate::layer& layer);
		virtual ~device_celularity();

		memory::const_empl::ptr const_emplace_cell() const;

	private:
		memory::const_empl::ptr m_const_cell = nullptr;

		void setup_const_memory(const innate::cell* c);

		
	};

}
