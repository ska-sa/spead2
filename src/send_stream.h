/**
 * @file
 */

#ifndef SPEAD_SEND_STREAM_H
#define SPEAD_SEND_STREAM_H

#include <atomic>
#include <functional>
#include <utility>
#include <vector>

namespace spead
{
namespace send
{

class writer;

template<typename T>
class countdown_handler
{
private:
    std::atomic<unsigned int> countdown;
    T handler;

public:
    countdown_handler(unsigned int count, T handler)
        : countdown(count), handler(handler)

    template<typename... Args>
    void operator()(Args&&... args)
    {
        if (--countdown == 0)
        {
            handler(args...);
        }
    }
};


typedef countdown_handler<std::function<void(void)> > write_handler;

class stream
{
private:
    std::vector<std::unique_ptr<writer> > writers;

public:
    template<typename Handler>
    void async_send_heap(const heap &h, Handler &&handler)
    {
        async_send_heap(h, std::function<void(void)>(std::forward<Handler>(handler)));
    }

    template<typename Handler>
    void async_send_heap(const heap &h, std::function<void(void)> handler)
    {
        // TODO: lock

        for (const auto w : writers)
        {
            w->async_send_heap
        }
    }
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_STREAM_H
