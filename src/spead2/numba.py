# Copyright 2021 National Research Foundation (SARAO)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Utility functions for writing callbacks using Numba."""

import numba.extending
from numba import types


@numba.extending.intrinsic
def intp_to_voidptr(typingctx, src):
    # The implementation is based on the example at
    # https://numba.readthedocs.io/en/stable/extending/high-level.html#implementing-intrinsics

    # check for accepted types
    if src in (types.intp, types.uintp):
        # create the expected type signature
        result_type = types.voidptr
        sig = result_type(src)

        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.inttoptr(src, llrtype)
        return sig, codegen


# The decorator doesn't preserve the docstring, so we have to assign it after
# the fact.
intp_to_voidptr.__doc__ = """
    Convert an integer (of type intp or uintp) to a void pointer.

    This is useful because numba doesn't (as of 0.54) support putting pointers
    into Records. They have to be smuggled in as intp, then converted to
    pointers with this function.
    """
