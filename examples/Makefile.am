# Copyright 2016 National Research Foundation (SARAO)
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

if SPEAD2_USE_CUDA

V_NVCC = $(v_NVCC_@AM_V@)
v_NVCC_ = $(v_NVCC_@AM_DEFAULT_V@)
v_NVCC_0 = @echo "  NVCC    " $@;
v_NVCC_1 =

V_NVCCLD = $(v_NVCCLD_@AM_V@)
v_NVCCLD_ = $(v_NVCCLD_@AM_DEFAULT_V@)
v_NVCCLD_0 = @echo "  NVCCLD  " $@;
v_NVCCLD_1 =

# These rules (and the defines above) are based on examining the generated
# Makefile.in and then modifying those rules to use NVCC instead of CXX.
.cu.o: $(top_srcdir)/include/spead2/*.h $(top_builddir)/include/spead2/common_features.h
	$(V_NVCC)$(NVCC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) -dc -o $@ $<

if SPEAD2_USE_IBV

gpudirect_example$(EXEEXT): $(gpudirect_example_OBJECTS) $(gpudirect_example_DEPENDENCIES) $(EXTRA_gpudirect_example_DEPENDENCIES)
	@rm -f gpudirect_example$(EXEEXT)
	$(V_NVCCLD)$(NVCC) $(AM_CXXFLAGS) $(CXXFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@ $(gpudirect_example_OBJECTS) $(gpudirect_example_LDADD) $(LIBS)

noinst_PROGRAMS += gpudirect_example
gpudirect_example_SOURCES = gpudirect_example.cu

endif

if SPEAD2_USE_GDRAPI

gdrapi_example$(EXEEXT): $(gdrapi_example_OBJECTS) $(gdrapi_example_DEPENDENCIES) $(EXTRA_gdrapi_example_DEPENDENCIES)
	@rm -f gdrapi_example$(EXEEXT)
	$(V_NVCCLD)$(NVCC) $(AM_CXXFLAGS) $(CXXFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@ $(gdrapi_example_OBJECTS) $(gdrapi_example_LDADD) $(LIBS)

noinst_PROGRAMS += gdrapi_example
gdrapi_example_SOURCES = gdrapi_example.cu
gdrapi_example_LDADD = -lgdrapi $(LDADD)

endif
endif
