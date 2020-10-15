/* Copyright 2019-2020 National Research Foundation (SARAO)
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
 */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>

#include <string>
#include <exception>
#include <system_error>
#include <dlfcn.h>
#include <spead2/common_loader_utils.h>
#include <spead2/common_logging.h>

namespace spead2
{

const char *loader_error_category::name() const noexcept
{
    return "loader";
}

std::string loader_error_category::message(int condition) const
{
    switch (loader_error(condition))
    {
        case loader_error::LIBRARY_ERROR:
            return "library could not be loaded";
        case loader_error::SYMBOL_ERROR:
            return "symbol could not be loaded";
        case loader_error::NO_INIT:
            return "loader_init was not called";
        default:
            return "unknown error";  // unreachable
    }
}

std::error_condition loader_error_category::default_error_condition(int condition) const noexcept
{
    switch (loader_error(condition))
    {
        case loader_error::LIBRARY_ERROR:
            return std::errc::no_such_file_or_directory;
        case loader_error::SYMBOL_ERROR:
            return std::errc::not_supported;
        case loader_error::NO_INIT:
            return std::errc::state_not_recoverable;
        default:
            return std::errc::not_supported;  // unreachable
    }
}

std::error_category &loader_category()
{
    static loader_error_category category;
    return category;
}

dl_handle::dl_handle(const char *filename)
{
    handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
    if (!handle)
    {
        std::error_code code((int) loader_error::LIBRARY_ERROR, loader_category());
        throw std::system_error(code, std::string("Could not open ") + filename + ": " + dlerror());
    }
}

dl_handle::~dl_handle()
{
    if (handle)
    {
        int ret = dlclose(handle);
        if (ret != 0)
            log_warning("dlclose failed: %s", dlerror());
    }
}

void *dl_handle::release()
{
    void *ret = handle;
    handle = nullptr;
    return ret;
}

void *dl_handle::sym(const char *name)
{
    void *ret = dlsym(handle, name);
    if (!ret)
    {
        std::error_code code((int) loader_error::SYMBOL_ERROR, loader_category());
        throw std::system_error(code, std::string("Symbol ") + name + " not found: " + dlerror());
    }
    return ret;
}

}  // namespace spead2
