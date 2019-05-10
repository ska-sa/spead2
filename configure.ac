# Copyright 2016, 2017, 2019 SKA South Africa
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

AC_INIT([spead2], m4_esyscmd([cut -d'"' -f2 spead2/_version.py | tr -d '\n']))
# Suppress the default CXXFLAGS
: ${CXXFLAGS=""}
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_SRCDIR([src/common_logging.cpp])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([foreign])
AC_PROG_CXX
AC_CHECK_TOOL(AR, ar, :)
AC_PROG_RANLIB
AC_ARG_VAR(AR, [Static library archiver command])
AC_ARG_VAR(RANLIB, [Generate index to static library])
AC_LANG(C++)

### Check for -std=c++11

save_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="-std=c++11 $CXXFLAGS"
SPEAD2_CHECK_FEATURE([compiler_cxx11], [-std=c++11], [unordered_map], [], [std::unordered_map<int, int> m], [],
                     [AC_MSG_ERROR([C++11 support is required])])
CXXFLAGS="$save_CXXFLAGS"

### Required libraries

SPEAD2_CHECK_LIB([boost/program_options.hpp], [boost_program_options],
                 [boost::program_options::option_description foo],
                 [], [AC_MSG_ERROR([boost_program_options is required])])
SPEAD2_CHECK_LIB([boost/system/system_error.hpp], [boost_system], [boost::system::error_code ec],
                 [], [AC_MSG_ERROR([boost_system is required])])
SPEAD2_CHECK_HEADER([boost/asio.hpp], [boost_system,pthread], [boost::asio::io_service io_service],
                    [], [AC_MSG_ERROR([boost_asio is required])])

### Optional libraries/features

SPEAD2_ARG_WITH(
    [ibv],
    [AS_HELP_STRING([--without-ibv], [Do not use libibverbs and librdmacm, even if detected])],
    [SPEAD2_USE_IBV],
    [SPEAD2_USE_IBV=1
     SPEAD2_CHECK_LIB(
        [infiniband/verbs.h], [ibverbs], [ibv_create_flow(NULL, NULL)],
        [], [SPEAD2_USE_IBV=0])
     SPEAD2_CHECK_LIB(
        [rdma/rdma_cma.h], [rdmacm], [rdma_create_event_channel()],
        [], [SPEAD2_USE_IBV=0])
    ]
)

SPEAD2_ARG_WITH(
    [ibv_exp],
    [AS_HELP_STRING([--without-ibv-exp], [Do not use verbs_exp.h, even if detected])],
    [SPEAD2_USE_IBV_EXP],
    [SPEAD2_USE_IBV_EXP=1
     SPEAD2_CHECK_HEADER(
        [infiniband/verbs_exp.h], [ibverbs], [ibv_exp_create_cq(NULL, 0, NULL, NULL, 0, NULL)],
        [], [SPEAD2_USE_IBV_EXP=0])
    ]
)

SPEAD2_ARG_WITH(
    [ibv_mprq],
    [AS_HELP_STRING([--without-ibv-mprq], [Do not use multi-packet receive queues, even if detected])],
    [SPEAD2_USE_IBV_MPRQ],
    [SPEAD2_USE_IBV_MPRQ=1
     SPEAD2_CHECK_FEATURE(
        [ibv_mprq], [multi-packet receive queue], [infiniband/verbs_exp.h], [ibverbs],
        [ibv_exp_cq_family_v1 *cq_intf;
         ibv_exp_wq_family *wq_intf;
         ibv_exp_qp_init_attr attr;
         ibv_exp_query_intf(NULL, NULL, NULL);
         ibv_exp_create_wq(NULL, NULL);
         ibv_exp_create_rwq_ind_table(NULL, NULL);],
        [], [SPEAD2_USE_IBV_MPRQ=0])
    ]
)

SPEAD2_ARG_WITH(
    [pcap],
    [AS_HELP_STRING([--without-pcap], [Do not use pcap, even if detected])],
    [SPEAD2_USE_PCAP],
    [SPEAD2_USE_PCAP=1
     SPEAD2_CHECK_HEADER(
        [pcap/pcap.h], [pcap], [pcap_open_offline(NULL, NULL)],
        [], [SPEAD2_USE_PCAP=0])
    ]
)


SPEAD2_ARG_WITH(
    [netmap],
    [AS_HELP_STRING([--without-netmap], [Do not use netmap, even if detected])],
    [SPEAD2_USE_NETMAP],
    [AC_LANG_PUSH(C)      # netmap_user.h doesn't do extern "C"
     SPEAD2_CHECK_FEATURE(
        [header_net_netmap_user_h], [netmap],
        [stddef.h net/netmap_user.h], [], [nm_open(NULL, NULL, 0, NULL)],
        [SPEAD2_USE_NETMAP=1], [], [#define NETMAP_WITH_LIBS]
     )
     AC_LANG_POP
    ]
)

SPEAD2_ARG_WITH(
    [recvmmsg],
    [AS_HELP_STRING([--without-recvmmsg], [Do not use recvmmsg system call])],
    [SPEAD2_USE_RECVMMSG],
    [AC_CHECK_FUNC([recvmmsg], [SPEAD2_USE_RECVMMSG=1], [])])

SPEAD2_ARG_WITH(
    [sendmmsg],
    [AS_HELP_STRING([--without-sendmmsg], [Do not use sendmmsg system call])],
    [SPEAD2_USE_SENDMMSG],
    [AC_CHECK_FUNC([sendmmsg], [SPEAD2_USE_SENDMMSG=1], [])])

SPEAD2_ARG_WITH(
    [eventfd],
    [AS_HELP_STRING([--without-eventfd], [Do not use eventfd system call for semaphores])],
    [SPEAD2_USE_EVENTFD],
    [AC_CHECK_FUNC([eventfd], [SPEAD2_USE_EVENTFD=1], [])])

SPEAD2_ARG_WITH(
    [pthread_setaffinity_np],
    [AS_HELP_STRING([--without-pthread_setaffinity_np], [Do not set thread affinity])],
    [SPEAD2_USE_PTHREAD_SETAFFINITY_NP],
    [SPEAD2_CHECK_FEATURE(
        [func_pthread_setaffinity_np],
        [pthread_setaffinity_np],
        [pthread.h],
        [pthread],
        [pthread_setaffinity_np(pthread_self(), 0, (cpu_set_t *) NULL)],
        [SPEAD2_USE_PTHREAD_SETAFFINITY_NP=1], []
    )]
)

SPEAD2_ARG_WITH(
    [movntdq],
    [AS_HELP_STRING([--without-movntdq], [Do not use MOVNTDQ instruction for non-temporal copies])],
    [SPEAD2_USE_MOVNTDQ],
    [SPEAD2_CHECK_FEATURE(
        [movntdq], [MOVNTDQ intrinsic], [emmintrin.h], [],
        [_mm_stream_si128((__m128i *) NULL, __m128i())],
        [SPEAD2_USE_MOVNTDQ=1], []
    )]
)

SPEAD2_ARG_WITH(
    [posix-semaphores],
    [AS_HELP_STRING([--without-posix-semaphores], [Do not POSIX semaphores, even if available])],
    [SPEAD2_USE_POSIX_SEMAPHORES],
    [SPEAD2_CHECK_FEATURE(
        [posix_semaphores], [POSIX semaphores],
        [unistd.h semaphore.h], [pthread],
        [#if defined(__APPLE__)
         # error "POSIX semaphores do not work on OS X"
         #endif
         #if !defined(_POSIX_SEMAPHORES) || _POSIX_SEMAPHORES < 0
         # error "POSIX semaphores marked unsupported"
         #endif
         sem_init(NULL, 0, 0)],
        [SPEAD2_USE_POSIX_SEMAPHORES=1], []
    )]
)

### Determine libraries to link against

LIBS="-lboost_system -lpthread -ldl"
AS_IF([test "x$SPEAD2_USE_IBV" = "x1"], [LIBS="$LIBS"])
AS_IF([test "x$SPEAD2_USE_PCAP" = "x1"], [LIBS="-lpcap $LIBS"])

### Build variants

AC_ARG_ENABLE(
    [debug-symbols],
    [AS_HELP_STRING([--enable-debug-symbols], [Build with debug symbols])])

AC_ARG_ENABLE(
    [debug-log],
    [AS_HELP_STRING([--enable-debug-log], [Emit debug-level log messages])])

AC_ARG_ENABLE(
    [coverage],
    [AS_HELP_STRING([--enable-coverage], [Build for coverage (gcov)])])

AC_ARG_ENABLE(
    [optimized],
    [AS_HELP_STRING([--disable-optimized], [Disable compiler optimizations])])

AC_ARG_ENABLE(
    [lto],
    [AS_HELP_STRING([--enable-lto], [Use link-time optimization])])

AM_CONDITIONAL([DEBUG_SYMBOLS], [test "x$enable_debug_symbols" = "xyes"])
AM_CONDITIONAL([DEBUG_LOG], [test "x$enable_debug_log" = "xyes"])
AM_CONDITIONAL([OPTIMIZED], [test "x$enable_optimized" != "xno"])
AM_CONDITIONAL([COVERAGE], [test "x$enable_coverage" = "xyes"])
AM_CONDITIONAL([LTO], [test "x$enable_lto" = "xyes"])
AM_CONDITIONAL([SPEAD2_USE_IBV], [test "x$SPEAD2_USE_IBV" = "x1"])
AM_CONDITIONAL([SPEAD2_USE_IBV_EXP], [test "x$SPEAD2_USE_IBV_EXP" = "x1"])
AM_CONDITIONAL([SPEAD2_USE_PCAP], [test "x$SPEAD2_USE_PCAP" = "x1"])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 examples/Makefile
                 include/Makefile
                 include/spead2/common_features.h
                 spead2.pc])
AC_OUTPUT
