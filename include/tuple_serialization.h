#pragma once

/*
 * tuple_serialization.h  -- Serialize tuples and pairs with Boost
 *
 * Copyright (C) 2015 Lorenz Hübschle-Schneider <lorenz@4z2.de>
 * Published under the Boost Software License, Version 1.0
 */

#include <tuple>
#include <utility>

// Boost serialization for tuples and pairs
namespace boost {
namespace serialization {

template<uint N>
struct tuple_serializer {
    template<class Archive, typename... Args>
    static inline void serialize(Archive& ar, std::tuple<Args...>& tuple)     {
        ar & std::get<N-1>(tuple);
        tuple_serializer<N-1>::serialize(ar, tuple);
    }
};

template<>
struct tuple_serializer<0> {
    template<class Archive, typename... Args>
    static inline void serialize(Archive&, std::tuple<Args...> &) {}
};

template<class Archive, typename... Args>
inline void serialize(Archive& ar, std::tuple<Args...>& t, const unsigned int) {
    tuple_serializer<std::tuple_size<std::tuple<Args...>>::value>::serialize(ar, t);
}

// Normal boost.serialize uses named thingies here
template<class Archive, typename T, typename U>
inline void serialize(Archive & ar, std::pair<T, U> & p, const unsigned int) {
    // remove const-ness to be able to load
    typedef typename boost::remove_const<T>::type typef;
    ar & const_cast<typef &>(p.first) ;
    ar & p.second ;
}

// Unwrap the templates. Yikes.
template <uint N, typename tuple>
struct is_tuple_bitwise_serializable
    : public mpl::and_<is_bitwise_serializable<typename std::tuple_element<N-1, tuple>::type>,
                       is_tuple_bitwise_serializable<N-1, tuple>> {
};

template<typename tuple>
struct is_tuple_bitwise_serializable<0, tuple> : public mpl::true_ {};

/// specialization of is_bitwise_serializable for tuples
template <typename... Args>
struct is_bitwise_serializable<std::tuple<Args...>>
    : public is_tuple_bitwise_serializable<
        std::tuple_size<std::tuple<Args...>>::value, std::tuple<Args...>> {};
} // namespace serialization
} // namespace boost
