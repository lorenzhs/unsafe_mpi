#pragma once

/*
 * allgatherv.hpp  -- MPI_Allgatherv wrapper for std::vector
 *
 * Copyright (C) 2015 Lorenz Hübschle-Schneider <lorenz@4z2.de>
 * Published under the Boost Software License, Version 1.0
 */

#include <errno.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <boost/mpi/packed_oarchive.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#include "common.h"

namespace unsafe_mpi {

template <typename T>
void allgatherv_serialize(const boost::mpi::communicator &comm, const std::vector<T> &in, std::vector<T> &out) {
    // Step 1: serialize input data
    boost::mpi::packed_oarchive oa(comm);
    if (!in.empty())
        oa << in;

    // Step 2: exchange sizes (archives' .size() is measured in bytes)
    // Need to cast to int because this is what MPI uses as size_t...
    const int in_size = static_cast<int>(in.size()),
        transmit_size = (in.empty() ? 0 : static_cast<int>(oa.size()));
    std::vector<int> in_sizes(comm.size()), transmit_sizes(comm.size());
    boost::mpi::all_gather(comm, in_size,       in_sizes.data());
    boost::mpi::all_gather(comm, transmit_size, transmit_sizes.data());

    // Step 3: calculate displacements from sizes (prefix sum)
    std::vector<int> displacements(comm.size() + 1);
    displacements[0] = sizeof(boost::mpi::packed_iarchive);
    for (int i = 1; i <= comm.size(); ++i) {
        displacements[i] = displacements[i-1] + transmit_sizes[i-1];
    }

    // Step 4: allocate space for result and MPI_Allgatherv
    char* recv = new char[displacements.back()];
    // If in.empty(), transmit_size is 0 so we don't really care
    auto sendptr = const_cast<void*>(oa.address());

    int status = MPI_Allgatherv(sendptr, transmit_size, MPI_PACKED, recv,
                                transmit_sizes.data(), displacements.data(),
                                MPI_PACKED, comm);

    if (status != 0) {
        ERR << "MPI_Allgatherv returned " << status << ", errno " << errno << std::endl;
        return;
    }

    // Step 5: deserialize received data
    // Preallocate storage to prevent reallocations
    std::vector<T> temp;
    size_t largest_size = *std::max_element(in_sizes.begin(), in_sizes.end());
    temp.reserve(largest_size);
    out.reserve(std::accumulate(in_sizes.begin(), in_sizes.end(), 0));

    // Deserialize archives one by one, inserting elements at the end of ̀out̀
    for (int i = 0; i < comm.size(); ++i) {
        if (in_sizes[i] == 0) {
            // We can ignore processes which didn't have anything to send
            continue;
        }
        boost::mpi::packed_iarchive archive(comm);
        archive.resize(transmit_sizes[i]);
        memcpy(archive.address(), recv + displacements[i], transmit_sizes[i]);

        temp.clear();
        temp.resize(in_sizes[i]);
        archive >> temp;
        out.insert(out.end(), temp.begin(), temp.end());
    }
}


template <typename T, typename transmit_type=uint64_t>
void allgatherv_unsafe(const boost::mpi::communicator &comm, const std::vector<T> &in, std::vector<T> &out) {
    static_assert((sizeof(T)/sizeof(transmit_type)) * sizeof(transmit_type) == sizeof(T),
        "Invalid transmit_type for element type (sizeof(transmit_type) is not a multiple of sizeof(T))");

    // Step 1: exchange sizes
    // We need to compute the displacement array, specifying for each PE
    // at which position in out to place the data received from it
    // Need to cast to int because this is what MPI uses as size_t...
    const int factor = sizeof(T) / sizeof(transmit_type);
    const int in_size = static_cast<int>(in.size()) * factor;
    std::vector<int> sizes(comm.size());
    boost::mpi::all_gather(comm, in_size, sizes.data());

    // Step 2: calculate displacements from sizes
    // Compute prefix sum to compute displacements from sizes
    std::vector<int> displacements(comm.size() + 1);
    displacements[0] = 0;
    std::partial_sum(sizes.begin(), sizes.end(), displacements.begin() + 1);
    // divide by factor by which T is larger than transmit_type
    out.resize(displacements.back() / factor);

    // Step 3: MPI_Allgatherv
    const transmit_type *sendptr = reinterpret_cast<const transmit_type*>(in.data());
    transmit_type *recvptr = reinterpret_cast<transmit_type*>(out.data());
    const MPI_Datatype datatype = boost::mpi::get_mpi_datatype<transmit_type>();

    int status = MPI_Allgatherv(sendptr, in_size, datatype, recvptr,
                                sizes.data(), displacements.data(),
                                datatype, comm);
    if (status != 0) {
        ERR << "MPI_Allgatherv returned " << status << ", errno " << errno << std::endl;
    }
}

template <typename T, typename transmit_type=uint64_t>
void allgatherv(const boost::mpi::communicator &comm, const std::vector<T> &in, std::vector<T> &out) {
    // Trivial (enough) datatypes can be transmit directly via MPI_Allgatherv
    // For all others, we have to serialize them using boost::serialize
    if (is_trivial_enough<T>::value) {
        allgatherv_unsafe<T, transmit_type>(comm, in, out);
    } else {
        allgatherv_serialize<T>(comm, in, out);
    }
}

}
