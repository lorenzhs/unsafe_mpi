#pragma once

/*
 * broadcast.hpp  -- MPI_Bcast wrapper for std::vector, supports serialization
 *                   using Boost.Serialize
 *
 * Copyright (C) 2015 Lorenz Hübschle-Schneider <lorenz@4z2.de>
 * Published under the Boost Software License, Version 1.0
 */


#include <mpi.h>
#include <errno.h>

#include <cstdint>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/datatype.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <boost/mpi/packed_oarchive.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include "common.h"

namespace unsafe_mpi {
template <typename T, typename transmit_type = uint64_t>
void broadcast(const boost::mpi::communicator &comm, std::vector<T> &data, int root) {
    const bool trivial = is_trivial_enough<T>::value;
    static_assert(!trivial || boost::mpi::is_mpi_datatype<T>() ||
        ((sizeof(T)/sizeof(transmit_type)) * sizeof(transmit_type) == sizeof(T)),
        "Invalid transmit_type for element type (sizeof(transmit_type) is not a multiple of sizeof(T))");

    if (comm.size() < 2) return;
    if (trivial) {
        // MPI only supports "int" as size type, and the MPI Forum's reply to
        // the issue can be summed up as "deal with it" (they refer to user-
        // defined contiguous data types, e.g. ones that hold 1024 elements)
        // MPI really hasn't moved on since the 90's...
        int size = static_cast<int>(data.size());
        // broadcast size and allocate space
        boost::mpi::broadcast<int>(comm, size, root);
        data.resize(size); // harmless on root, required on others
        // broadcast elements as transmit_type
        auto ptr = reinterpret_cast<transmit_type*>(data.data());
        size = static_cast<int>(size * sizeof(T)/sizeof(transmit_type));
        boost::mpi::broadcast(comm, ptr, size, root);
    } else if (boost::mpi::is_mpi_datatype<T>()) {
        // We can use Boost.MPI directly to transmit MPI datatypes
        // But send size and data separately to avoid vector serialization
        // I don't think this codepath will ever be called, as all
        // native MPI datatypes should be trivial enough.
        int size = static_cast<int>(data.size());
        boost::mpi::broadcast(comm, size, root);
        data.resize(size); // harmless on root, required on others
        boost::mpi::broadcast<T>(comm, data.data(), size, root);
    } else {
        // Boost.MPI doesn't use MPI_Broadcast for types it doesn't know. WTF.
        // Therefore, we need to do the archive broadcast ourselves.
        if (comm.rank() == root) {
            // Serialize data
            boost::mpi::packed_oarchive oa(comm);
            oa << data;
            // Broadcast archive size
            size_t archive_size = oa.size();
            boost::mpi::broadcast(comm, archive_size, root);
            // Broadcast archive data
            auto sendptr = const_cast<void*>(oa.address());
            int status = MPI_Bcast(sendptr, static_cast<int>(archive_size),
                                   MPI_PACKED, root, comm);
            if (status != 0) {
                ERR << "MPI_Bcast returned non-zero value " << status
                    << ", errno: " << errno << std::endl;
            }
        } else {
            // Receive archive size and allocate space
            size_t archive_size;
            boost::mpi::broadcast(comm, archive_size, root);
            boost::mpi::packed_iarchive ia(comm);
            ia.resize(archive_size);
            // Receive broadcast archive data
            auto recvptr = ia.address();
            int status = MPI_Bcast(recvptr, static_cast<int>(archive_size),
                                   MPI_PACKED, root, comm);
            if (status != 0) {
                ERR << "MPI_Bcast returned non-zero value " << status
                    << ", errno: " << errno << std::endl;
                return;
            }
            // Unpack received data
            ia >> data;
        }
    }
}
}
