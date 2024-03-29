/* Copyright 2015, 2023 National Research Foundation (SARAO)
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Logging facilities
 */

#include <spead2/common_logging.h>
#include <iostream>
#include <cassert>
#include <system_error>
#include <cerrno>

using namespace std::literals;

namespace spead2
{

static const char * const level_names[] =
{
    "warning",
    "info",
    "debug"
};

std::ostream &operator<<(std::ostream &o, log_level level)
{
    unsigned int level_idx = static_cast<unsigned int>(level);
    assert(level_idx < sizeof(level_names) / sizeof(level_names[0]));
    return o << level_names[level_idx];
}

static void default_log_function(log_level level, const std::string &msg)
{
    std::cerr << "spead2: " << level << ": " << msg << "\n";
}

static std::function<void(log_level, const std::string &)> log_function = default_log_function;

std::function<void(log_level, const std::string &)>
set_log_function(std::function<void(log_level, const std::string &)> f)
{
    auto old = log_function;
    log_function = f;
    return old;
}

namespace detail
{

void log_msg_impl(log_level level, const std::string &msg)
{
    log_function(level, msg);
}

} // namespace detail

void log_errno(const char *format, int err)
{
    std::error_code code(err, std::system_category());
    log_warning(format, code.value(), code.message());
}

void log_errno(const char *format)
{
    log_errno(format, errno);
}

[[noreturn]] void throw_errno(const char *msg, int err)
{
    if (err == 0)
    {
        throw std::system_error(
            std::make_error_code(std::errc::invalid_argument),
            msg + " (unknown error)"s);
    }
    else
    {
        std::system_error exception(err, std::system_category(), msg);
        throw exception;
    }
}

[[noreturn]] void throw_errno(const char *msg)
{
    throw_errno(msg, errno);
}

} // namespace spead2
