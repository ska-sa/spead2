# Copyright 2016, 2017, 2019-2020 National Research Foundation (SARAO)
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

AC_INIT([spead2], m4_esyscmd([python3 gen/get_version.py full]))
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

# pkg-config files for libibverbs are quite new, so fall back to searching
# system paths.
IBV_CFLAGS=""
SPEAD2_ARG_WITH(
    [ibv],
    [AS_HELP_STRING([--without-ibv], [Do not use libibverbs and librdmacm, even if detected])],
    [SPEAD2_USE_IBV],
    [PKG_CHECK_MODULES([IBV], [libibverbs librdmacm], [SPEAD2_USE_IBV=1],
        [SPEAD2_CHECK_LIB(
         [infiniband/verbs.h], [ibverbs], [ibv_create_wq(NULL, NULL)],
         [SPEAD2_USE_IBV=1
         ], [:])
        ])
    ]
)

MLX5DV_CFLAGS=""
SPEAD2_ARG_WITH(
    [mlx5dv],
    [AS_HELP_STRING([--without-mlx5dv], [Do not use mlx5dv.h, even if detected])],
    [SPEAD2_USE_MLX5DV],
    [PKG_CHECK_MODULES(MLX5DV, [libmlx5],
        [SPEAD2_CHECK_HEADER(
            [infiniband/mlx5dv.h], [], [mlx5dv_create_cq(NULL, NULL, NULL)],
            [SPEAD2_USE_MLX5DV=1], [], [], ["$MLX5DV_CFLAGS"], ["$MLX5DV_LIBS"])
        ],
        [SPEAD2_CHECK_HEADER(
            [infiniband/mlx5dv.h], [mlx5], [mlx5dv_create_cq(NULL, NULL, NULL)],
            [SPEAD2_USE_MLX5DV=1], [])
        ])
    ]
)
AS_IF([test "x$SPEAD2_USE_IBV" = "x0"], [SPEAD2_USE_MLX5DV=0])

SPEAD2_ARG_WITH(
    [ibv-hw-rate-limit],
    [AS_HELP_STRING([--without-ibv-hw-rate-limit], [Disable ibverbs hardware rate limiting])],
    [SPEAD2_USE_IBV_HW_RATE_LIMIT],
    [SPEAD2_CHECK_FEATURE(
        [ibv_modify_qp_rate_limit],
        [ibv_modify_qp_rate_limit],
        [infiniband/verbs.h], [],
        [ibv_modify_qp_rate_limit(NULL, NULL)],
        [SPEAD2_USE_IBV_HW_RATE_LIMIT=1], [],
        [], ["$IBV_CFLAGS"], [])
    ]
)
AS_IF([test "x$SPEAD2_USE_IBV" = "x0"], [SPEAD2_USE_IBV_HW_RATE_LIMIT=0])

SPEAD2_ARG_WITH(
    [pcap],
    [AS_HELP_STRING([--without-pcap], [Do not use pcap, even if detected])],
    [SPEAD2_USE_PCAP],
    [PKG_CHECK_MODULES([PCAP], [libpcap], [SPEAD2_USE_PCAP=1],
        [SPEAD2_CHECK_HEADER(
            [pcap/pcap.h], [pcap], [pcap_open_offline(NULL, NULL)],
            [SPEAD2_USE_PCAP=1
             PCAP_CFLAGS=""
             PCAP_LIBS="-lpcap"
            ],
            [])
        ])
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
        [cpu_set_t cpuset = {};
         pthread_setaffinity_np(pthread_self(), 0, &cpuset)],
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
    [AS_HELP_STRING([--without-posix-semaphores], [Do not use POSIX semaphores, even if available])],
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
         sem_t sem;
         sem_init(&sem, 0, 0)],
        [SPEAD2_USE_POSIX_SEMAPHORES=1], []
    )]
)

SPEAD2_ARG_WITH(
    [cuda],
    [AS_HELP_STRING([--without-cuda], [Do not build CUDA examples])],
    [SPEAD2_USE_CUDA],
    [AC_CHECK_PROG([NVCC], [nvcc], [nvcc])
     AS_IF([test "x$NVCC" != "x"], [SPEAD2_USE_CUDA=1])]
)
AC_ARG_VAR(NVCC)

SPEAD2_ARG_WITH(
    [gdrapi],
    [AS_HELP_STRING([--without-gdrapi], [Do not build examples using gdrcopy])],
    [SPEAD2_USE_GDRAPI],
    [SPEAD2_CHECK_HEADER(
        [gdrapi.h], [gdrapi], [gdr_open()],
        [SPEAD2_USE_GDRAPI=1])
    ])

### Determine libraries to link against and include paths

LIBS="-lboost_system -lpthread -ldl"
SPEAD2_CFLAGS=""
AS_IF([test "x$SPEAD2_USE_PCAP" = "x1"], [
    LIBS="$PCAP_LIBS $LIBS"
    SPEAD2_CFLAGS="$PCAP_CFLAGS $SPEAD2_CFLAGS"
])
AS_IF([test "x$SPEAD2_USE_IBV" = "x1"], [
    SPEAD2_CFLAGS="$IBV_CFLAGS $SPEAD2_CFLAGS"
])
AS_IF([test "x$SPEAD2_USE_MLX5DV" = "x1"], [
    SPEAD2_CFLAGS="$MLX5DV_CFLAGS $SPEAD2_CFLAGS"
])
CXXFLAGS="$SPEAD2_CFLAGS $CXXFLAGS"
AC_SUBST(SPEAD2_CFLAGS)

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
AM_CONDITIONAL([SPEAD2_USE_CUDA], [test "x$SPEAD2_USE_CUDA" = "x1"])
AM_CONDITIONAL([SPEAD2_USE_GDRAPI], [test "x$SPEAD2_USE_GDRAPI" = "x1"])

AC_SUBST(SPEAD2_MAJOR, m4_esyscmd([python3 gen/get_version.py major]))
AC_SUBST(SPEAD2_MINOR, m4_esyscmd([python3 gen/get_version.py minor]))
AC_SUBST(SPEAD2_PATCH, m4_esyscmd([python3 gen/get_version.py patch]))
AC_SUBST(SPEAD2_VERSION, m4_esyscmd([python3 gen/get_version.py full]))

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 examples/Makefile
                 include/Makefile
                 include/spead2/common_features.h
                 spead2.pc
                 python-build.cfg])
AC_OUTPUT

echo "Compilation options:"
echo ""
SPEAD2_PRINT_CONDITION([debug symbols], [DEBUG_SYMBOLS])
SPEAD2_PRINT_CONDITION([debug logging], [DEBUG_LOG])
SPEAD2_PRINT_CONDITION([compiler optimization], [OPTIMIZED])
SPEAD2_PRINT_CONDITION([link-time optimization], [LTO])
SPEAD2_PRINT_CONDITION([coverage], [COVERAGE])
SPEAD2_PRINT_FEATURE([MOVNTDQ instruction], [test "x$SPEAD2_USE_MOVNTDQ" = "x1"])
echo ""
echo "System calls:"
echo ""
SPEAD2_PRINT_FEATURE([recvmmsg], [test "x$SPEAD2_USE_RECVMMSG" = "x1"])
SPEAD2_PRINT_FEATURE([sendmmsg], [test "x$SPEAD2_USE_SENDMMSG" = "x1"])
SPEAD2_PRINT_FEATURE([eventfd], [test "x$SPEAD2_USE_EVENTFD" = "x1"])
SPEAD2_PRINT_FEATURE([POSIX semaphores], [test "x$SPEAD2_USE_POSIX_SEMAPHORES" = "x1"])
echo ""
echo "Libraries:"
echo ""
SPEAD2_PRINT_FEATURE([ibverbs], [test "x$SPEAD2_USE_IBV" = "x1"])
SPEAD2_PRINT_FEATURE([ibverbs HW rate limit], [test "x$SPEAD2_USE_IBV_HW_RATE_LIMIT" = "x1"])
SPEAD2_PRINT_FEATURE([mlx5dv], [test "x$SPEAD2_USE_MLX5DV" = "x1"])
SPEAD2_PRINT_FEATURE([pcap], [test "x$SPEAD2_USE_PCAP" = "x1"])
SPEAD2_PRINT_FEATURE([CUDA (examples only)], [test "x$SPEAD2_USE_CUDA" = "x1"])
SPEAD2_PRINT_FEATURE([gdrcopy (examples only)], [test "x$SPEAD2_USE_GDRAPI" = "x1"])
echo ""
