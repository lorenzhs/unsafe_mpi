#pragma once

/*
 * common.hpp  -- Triviality test for data types
 *
 * Copyright (C) 2015 Lorenz Hübschle-Schneider <lorenz@4z2.de>
 * Published under the Boost Software License, Version 1.0
 */

#include <type_traits>
#include <utility>

namespace unsafe_mpi {

/*
 * Define "trivial enough" type trait for things that can be copied bitwise
 * mostly because std::is_trivial is very strict, and std::pair does not fulfill
 * its requirements, but is trivial enough [TM] in practice.
 *
 * Override this for any sufficiently trivial own data types.
 */
template <typename T>
struct is_trivial_enough : public std::is_trivial<T> {};

// Pairs are trivial enough [TM] if both components are.
template <typename U, typename V>
struct is_trivial_enough<std::pair<U,V>> :
    public std::integral_constant<bool,
        is_trivial_enough<U>::value && is_trivial_enough<V>::value
    > {};

}
