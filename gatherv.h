#pragma once

#include <errno.h>
#include <mpi.h>

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

template <typename T, typename transmit_type = uint64_t>
void gatherv_trivial(const boost::mpi::communicator &comm,
                            const std::vector<T> &in, std::vector<T> &out,
                            const int root) {

    static_assert((sizeof(T)/sizeof(transmit_type)) * sizeof(transmit_type) == sizeof(T),
                  "Invalid transmit_type for element type (sizeof(transmit_type) is not a multiple of sizeof(T))");

    // exchange sizes
    const int factor = sizeof(T) / sizeof(transmit_type);
    const int sendsize = static_cast<int>(in.size() * factor);

    const auto datatype = boost::mpi::get_mpi_datatype<transmit_type>();
    const auto sendptr = reinterpret_cast<const transmit_type*>(in.data());

    if (comm.rank() == root) {
        // Receive sizes
        std::vector<int> sizes;
        sizes.reserve(comm.size());
        boost::mpi::gather(comm, sendsize, sizes, root);

        // Calculate displacements from spaces
        std::vector<int> displacements(sizes.size() + 1);
        std::partial_sum(sizes.begin(), sizes.end(), displacements.begin() + 1);
        const int outsize = displacements.back();
        // Allocate space
        // in terms of #elements -> divide by factor
        out.resize(outsize / factor);

        auto recvptr = reinterpret_cast<transmit_type*>(out.data());
        MPI_Gatherv(sendptr, sendsize, datatype,
                    recvptr, sizes.data(), displacements.data(),
                    datatype, root, comm);
    } else {
        // send size, then gather
        boost::mpi::gather(comm, sendsize, root);
        MPI_Gatherv(sendptr, sendsize, datatype,
                    nullptr, nullptr, nullptr,
                    datatype, root, comm);
    }
}

// UNTESTED, mostly copied from allgatherv
template <typename T>
void gatherv_serialize(const boost::mpi::communicator &comm, const std::vector<T> &in, std::vector<T> &out, const int root) {
    // Step 1: serialize input data
    boost::mpi::packed_oarchive oa(comm);
    if (!in.empty())
        oa << in;

    // Step 2: exchange sizes (archives' .size() is measured in bytes)
    // Need to cast to int because this is what MPI uses as size_t...
    const int in_size = static_cast<int>(in.size()),
        transmit_size = (in.empty() ? 0 : static_cast<int>(oa.size()));
    // If in.empty(), transmit_size is 0 so we don't really care
    auto sendptr = const_cast<void*>(oa.address());

    if (comm.rank() == root) {
        std::vector<int> in_sizes(comm.size()), transmit_sizes(comm.size());
        boost::mpi::gather<int>(comm, in_size,       in_sizes.data(), root);
        boost::mpi::gather<int>(comm, transmit_size, transmit_sizes.data(), root);

        // Step 3: calculate displacements from sizes (prefix sum)
        std::vector<int> displacements(comm.size() + 1);
        displacements[0] = sizeof(boost::mpi::packed_iarchive);
        for (int i = 1; i <= comm.size(); ++i) {
            displacements[i] = displacements[i-1] + transmit_sizes[i-1];
        }

        // Step 4: allocate space for result and MPI_Allgatherv
        char* recv = new char[displacements.back()];

        int status = MPI_Gatherv(sendptr, transmit_size, MPI_PACKED, recv,
                                 transmit_sizes.data(), displacements.data(),
                                 MPI_PACKED, root, comm);

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

    } else {
        boost::mpi::gather<int>(comm, in_size, root);
        boost::mpi::gather<int>(comm, transmit_size, root);

        int status = MPI_Gatherv(sendptr, transmit_size, MPI_PACKED,
                                 nullptr, nullptr, nullptr,
                                 MPI_PACKED, root, comm);
        if (status != 0) {
            ERR << "MPI_Allgatherv returned " << status << ", errno " << errno << std::endl;
            return;
        }


    }

}


template <typename T, typename transmit_type = uint64_t>
void gatherv(const boost::mpi::communicator &comm, const std::vector<T> &in, std::vector<T> &out, const int root) {
    if (is_trivial_enough<T>::value) {
        gatherv_trivial<T, transmit_type>(comm, in, out, root);
    } else {
        gatherv_serialize<T>(comm, in, out, root);
    }
}

}
