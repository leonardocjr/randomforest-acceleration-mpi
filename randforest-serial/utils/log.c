/*
 * Modifications by Sermet Pekin , 19.09.2025 :
    * - Added log_if_level function for conditional logging based on log_level.
 */

#include "log.h"
#include <stdio.h>
#include <stdarg.h>
#include "utils.h"


void log_if_level(int level, const char *format, ...) {
    if (log_level > level) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}   