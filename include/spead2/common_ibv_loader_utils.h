/* Copyright 2019 SKA South Africa
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

#ifndef SPEAD2_COMMON_IBV_LOADER_UTILS_H
#define SPEAD2_COMMON_IBV_LOADER_UTILS_H

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <spead2/common_features.h>

#if SPEAD2_USE_IBV

#include <string>
#include <system_error>
#include <exception>

namespace spead2
{

enum class ibv_loader_error : int
{
    LIBRARY_ERROR,
    SYMBOL_ERROR,
    NO_INIT
};

class ibv_loader_error_category : public std::error_category
{
public:
    virtual const char *name() const noexcept override;
    virtual std::string message(int condition) const override;
    virtual std::error_condition default_error_condition(int condition) const noexcept override;
};

std::error_category &ibv_loader_category();

[[noreturn]] void ibv_loader_stub(std::exception_ptr init_result);

class dl_handle
{
private:
    void *handle;
public:
    dl_handle(const char *filename);
    ~dl_handle();

    void *sym(const char *name);
    void *release();
};

} // namespace spead2

#endif // SPEAD2_USE_IBV
#endif // SPEAD2_COMMON_IBV_LOADER_UTILS_H
