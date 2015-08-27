#pragma once

/*
 * Fast, but potentially unsafe additions to Boost.MPI
 *
 * Copyright (C) 2015 Lorenz Hübschle-Schneider <lorenz@4z2.de>
 * Published under the Boost Software License, Version 1.0
 */

// Point-to-Point communication
#include "include/point-to-point.h"

// Collective communication
#include "include/broadcast.h"
#include "include/allgatherv.h"
#include "include/gatherv.h"
