/**
 * @file
 *
 * Logging facilities
 */

#include "common_logging.h"
#include <iostream>
#include <cassert>

namespace spead2
{

static const char * const level_names[] =
{
    "warning",
    "info",
    "debug"
};

static void default_log_function(log_level level, const std::string &msg)
{
    unsigned int level_idx = static_cast<unsigned int>(level);
    assert(level_idx < sizeof(level_names) / sizeof(level_names[0]));
    std::cerr << "spead2: " << level_names[level_idx] << ": " << msg << "\n";
}

static std::function<void(log_level, const std::string &)> log_function = default_log_function;

void set_log_function(std::function<void(log_level, const std::string &)> f)
{
    log_function = f;
}

namespace detail
{

void log_msg_impl(log_level level, const std::string &msg)
{
    log_function(level, msg);
}

} // namespace detail

} // namespace spead2
