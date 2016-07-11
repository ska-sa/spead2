AC_INIT([spead2], [0.11.0])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE([foreign])
AC_PROG_CXX
AC_PROG_RANLIB
AC_LANG(C++)

### Required libraries

SPEAD2_CHECK_LIB([boost/program_options.hpp], [boost_program_options],
                 [boost::program_options::option_description foo],
                 [], [AC_MSG_ERROR([boost_program_options is required])])
SPEAD2_CHECK_LIB([boost/system/system_error.hpp], [boost_system], [boost::system::error_code ec],
                 [], [AC_MSG_ERROR([boost_system is required])])
SPEAD2_CHECK_HEADER([boost/asio.hpp], [boost_system], [boost::asio::io_service io_service],
                    [], [AC_MSG_ERROR([boost_asio is required])])

### Optional libraries/features

SPEAD2_ARG_WITH([ibv], [Use libibverbs and librdmacm], [SPEAD2_USE_IBV],
    [SPEAD2_USE_IBV=1
     SPEAD2_CHECK_LIB(
        [infiniband/verbs.h], [ibverbs], [ibv_create_flow(NULL, NULL)],
        [], [SPEAD2_USE_IBV=0])
     SPEAD2_CHECK_LIB(
        [rdma/rdma_cma.h], [rdmacm], [rdma_create_event_channel()],
        [], [SPEAD2_USE_IBV=0])
    ]
)

SPEAD2_ARG_WITH([netmap], [Use netmap], [SPEAD2_USE_NETMAP],
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
    [recvmmsg], [Use recvmmsg system call], [SPEAD2_USE_RECVMMSG],
    [AC_CHECK_FUNC([recvmmsg], [SPEAD2_USE_RECVMMSG=1], [])])

SPEAD2_ARG_WITH(
    [eventfd], [Use eventfd system call for semaphores], [SPEAD2_USE_EVENTFD],
    [AC_CHECK_FUNC([eventfd], [SPEAD2_USE_EVENTFD=1], [])])

SPEAD2_ARG_WITH(
    [pthread_setaffinity_np], [Support setting thread affinity],
    [SPEAD2_USE_PTHREAD_SETAFFINITY_NP],
    [SPEAD2_CHECK_FEATURE(
        [func_pthread_setaffinity_np],
        [pthread_setaffinity_np],
        [pthread.h],
        [pthread],
        [pthread_setaffinity_np(pthread_self(), 0, NULL)],
        [SPEAD2_USE_PTHREAD_SETAFFINITY_NP=1], []
    )]
)

SPEAD2_ARG_WITH(
    [movntdq], [Use MOVNTDQ instruction for non-temporal copies],
    [SPEAD2_USE_MOVNTDQ],
    [SPEAD2_CHECK_FEATURE(
        [movntdq], [MOVNTDQ intrinsic], [emmintrin.h], [],
        [_mm_stream_si128((__m128i *) NULL, __m128i())],
        [SPEAD2_USE_MOVNTDQ=1], []
    )]
)

SPEAD2_ARG_WITH(
    [posix-semaphores], [Use POSIX semaphores],
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

AM_CONDITIONAL([SPEAD2_USE_IBV], [test "x$SPEAD2_USE_IBV" = "x1"])

AC_CONFIG_FILES([Makefile src/Makefile src/common_features.h spead2/_version.py])
AC_OUTPUT
