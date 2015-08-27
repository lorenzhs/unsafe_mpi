#pragma once

#include <cstdint>
#include <vector>

#include <boost/mpi/communicator.hpp>

#include "common.h"

namespace unsafe_mpi {

// Send `size` elements of type `T` starting at `data` to `dest` via `comm` with `tag`,
// using trivial type `transmit_type` if `T` is Standard Layout
template <typename T, typename transmit_type = uint64_t>
static void send(const boost::mpi::communicator &comm, int dest, int tag, const T *data, const size_t size) {
    const bool trivial = is_trivial_enough<T>::value;
    static_assert(!trivial || (sizeof(T)/sizeof(transmit_type)) * sizeof(transmit_type) == sizeof(T),
                  "Invalid transmit_type for element type (sizeof(transmit_type) is not a multiple of sizeof(T))");

    // send size
    comm.send(dest, tag, size);

    // send actual data
    if (trivial) {
        auto sendptr = reinterpret_cast<const transmit_type*>(data);
        auto sendsize = size * sizeof(T)/sizeof(transmit_type);
        comm.send(dest, tag, sendptr, sendsize);
    } else {
        comm.send(dest, tag, data, size);
    }
}


// convenience wrapper for vectors
template <typename T, typename transmit_type = uint64_t>
static void send(const boost::mpi::communicator &comm, int dest, int tag, const std::vector<T> &data) {
    send<T, transmit_type>(comm, dest, tag, data.data(), data.size());
}


template <typename T, typename transmit_type = uint64_t>
static void recv(const boost::mpi::communicator &comm, int src, int tag, std::vector<T> &data) {
    const bool trivial = is_trivial_enough<T>::value;
    static_assert(!trivial || (sizeof(T)/sizeof(transmit_type)) * sizeof(transmit_type) == sizeof(T),
                  "Invalid transmit_type for element type (sizeof(transmit_type) is not a multiple of sizeof(T))");

    auto size = data.size(); // for the type deduction
    // receive size and resize
    comm.recv(src, tag, size);
    data.resize(size);

    // receive actual data
    if (trivial) {
        auto recvptr = reinterpret_cast<transmit_type*>(data.data());
        auto recvsize = size * sizeof(T)/sizeof(transmit_type);
        comm.recv(src, tag, recvptr, recvsize);
    } else {
        comm.recv(src, tag, data.data(), size);
    }
}

}
