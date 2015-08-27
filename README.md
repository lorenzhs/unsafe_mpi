## unsafe_mpi: unsafe but fast additions to Boost.MPI

This is a small C++11 library that wraps several MPI operations in order to expose them more efficiently than the (otherwise excellent) [Boost.MPI](http://www.boost.org/libs/mpi) does.

Two aspects are covered at the moment:
- Boost.MPI always falls back to point-to-point communication operations for data types that it needs to serialize (using Boost.Serialize), which is quite wasteful. This library implements some of these operations, more may be added when I need them (contributions welcome!)
- Some data types are trivial enough not to require serialization, but Boost.MPI serializes them nonetheless. For instance, `std::pair<T1, T2>` is trivial enough [TM] to copy bitwise if both `T1` and `T2` are. In the same vein, we do not need to serialize `std::vector<T>` for data types that are trivial enough, but can just transmit its size and then its raw data. We thus `reinterpret_cast<>` them to an MPI Datatype (`uint64_t` by default, but you may require smaller a smaller type for, e.g., `std::pair<char, char>`) and transmit them as such.

There are a bunch of scenarios where these things might go wrong, but I think the name `unsafe_mpi` conveys this fairly well. It's also not properly tested, making it even less safe to use ;)

Published under the Boost Software License, Version 1.0 (see LICENSE)
