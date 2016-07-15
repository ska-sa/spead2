# Copyright 2016 SKA South Africa
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

include $(top_srcdir)/src/Makefile.inc.am

noinst_PROGRAMS = test_recv test_send test_ringbuffer

test_recv_SOURCES = test_recv.cpp

test_send_SOURCES = test_send.cpp

test_ringbuffer_SOURCES = test_ringbuffer.cpp
test_ringbuffer_LDADD = -lboost_program_options $(LDADD)
