include $(top_srcdir)/src/Makefile.inc.am

noinst_PROGRAMS = test_recv test_send test_ringbuffer

test_recv_SOURCES = test_recv.cpp

test_send_SOURCES = test_send.cpp

test_ringbuffer_SOURCES = test_ringbuffer.cpp
test_ringbuffer_LDADD = -lboost_program_options $(LDADD)
