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

#include <pybind11/pybind11.h>
#include <spead2/py_common.h>
#include <spead2/py_recv.h>
#include <spead2/py_send.h>

namespace py = pybind11;

PYBIND11_MODULE(_spead2, m)
{
    spead2::register_module(m);
    spead2::recv::register_module(m);
    spead2::send::register_module(m);
    spead2::register_logging();
    spead2::register_atexit();
}
