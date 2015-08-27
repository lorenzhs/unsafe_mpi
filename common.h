#pragma once

#include <type_traits>
#include <utility>

namespace unsafe_mpi {

// Sort-of unsafe MPI operations on "trivial enough" (i.e. standard layout) data
// mostly because std::pair isn't technically a trivial type, but we want to treat it like one

template <typename T>
struct is_trivial_enough : public std::is_trivial<T> {};

// Pairs are trivial enough [TM] if both components are trivial enough [TM]
template <typename U, typename V>
struct is_trivial_enough<std::pair<U,V>> :
    public std::integral_constant<bool,
        is_trivial_enough<U>::value && is_trivial_enough<V>::value
    > {};



}
