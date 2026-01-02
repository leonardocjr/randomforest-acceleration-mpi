/*
 * Modifications by Sermet Pekin , 19.09.2025 :
    * - Added log_if_level function for conditional logging based on log_level.
 */

#ifndef log_h
#define log_h

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "utils.h"
void log_if_level(int level, const char *format, ...);


#endif // log_h
