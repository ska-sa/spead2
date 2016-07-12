/* Copyright 2015 SKA South Africa
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

#ifndef SPEAD2_COMMON_LOGGING_H
#define SPEAD2_COMMON_LOGGING_H

#include <functional>
#include <boost/format.hpp>
#include <boost/preprocessor/cat.hpp>
#include <spead2/common_defines.h>

namespace spead2
{

enum class log_level : unsigned int
{
    warning = 0,
    info = 1,
    debug = 2
};

namespace detail
{

void log_msg_impl(log_level level, const std::string &msg);

static inline void apply_format(boost::format &format)
{
}

template<typename T0, typename... Ts>
static inline void apply_format(boost::format &format, T0&& arg0, Ts&&... args)
{
    format % std::forward<T0>(arg0);
    apply_format(format, std::forward<Ts>(args)...);
}

} // namespace detail

void set_log_function(std::function<void(log_level, const std::string &)>);

/**
 * Log a plain string at a given log level. Do not append a final newline to @a msg.
 */
static inline void log_msg(log_level level, const std::string &msg)
{
    if (level <= SPEAD2_MAX_LOG_LEVEL)
        detail::log_msg_impl(level, msg);
}

static inline void log_msg(log_level level, const char *msg)
{
    if (level <= SPEAD2_MAX_LOG_LEVEL)
        detail::log_msg_impl(level, msg);
}

/**
 * Log a message where the arguments are processed by @c boost::format.
 */
template<typename T0, typename... Ts>
static inline void log_msg(log_level level, const char *format, T0&& arg0, Ts&&... args)
{
    if (level <= SPEAD2_MAX_LOG_LEVEL)
    {
        boost::format formatter(format);
        detail::apply_format(formatter, std::forward<T0>(arg0), std::forward<Ts>(args)...);
        detail::log_msg_impl(level, formatter.str());
    }
}

#define SPEAD2_DEFINE_LOG_LEVEL(name) \
    static inline void BOOST_PP_CAT(log_, name)(const std::string &msg) \
    {                                                                  \
        log_msg(log_level::name, msg);                                 \
    }                                                                  \
    static inline void BOOST_PP_CAT(log_, name)(const char *msg)       \
    {                                                                  \
        log_msg(log_level::name, msg);                                 \
    }                                                                  \
    template<typename T0, typename... Ts>                              \
    static inline void BOOST_PP_CAT(log_, name)(const char *format, T0 &&arg0, Ts&&... args) \
    {                                                                  \
        log_msg(log_level::name, format, std::forward<T0>(arg0), std::forward<Ts>(args)...); \
    }

SPEAD2_DEFINE_LOG_LEVEL(debug)
SPEAD2_DEFINE_LOG_LEVEL(info)
SPEAD2_DEFINE_LOG_LEVEL(warning)

#undef SPEAD2_DEFINE_LOG_LEVEL

/// Write a warning log message about an errno constant
void log_errno(const char *format, int err);
/// Write a warning log message about errno
void log_errno(const char *format);

/// Throw a system_error about an errno constant
[[noreturn]] void throw_errno(const char *msg, int err);
/// Throw a system_error about errno
[[noreturn]] void throw_errno(const char *msg);

} // namespace spead2

#endif // SPEAD2_COMMON_LOGGING_H
