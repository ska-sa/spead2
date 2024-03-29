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
 */

#ifndef SPEAD2_RECV_MEM_H
#define SPEAD2_RECV_MEM_H

#include <cstdint>
#include <spead2/recv_stream.h>

namespace spead2::recv
{

class reader;

/**
 * Reader class that feeds data from a memory buffer to a stream. The caller
 * must ensure that the underlying memory buffer is not destroyed before
 * this class.
 *
 * @note For simple cases, use @ref mem_to_stream instead. This class is
 * only necessary if one wants to plug in to a @ref stream.
 */
class mem_reader : public reader
{
private:
    /// Start of data
    const std::uint8_t *ptr;
    /// Length of data
    std::size_t length;

public:
    mem_reader(stream &owner,
               const std::uint8_t *ptr, std::size_t length);

    virtual void start() override;
    virtual bool lossy() const override;
};

} // namespace spead2::recv

#endif // SPEAD2_RECV_MEM_H
