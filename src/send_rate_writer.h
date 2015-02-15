/**
 * @file
 */

#ifndef SPEAD_SEND_RATE_WRITER_H
#define SPEAD_SEND_RATE_WRITER_H

#include <memory>
#include <utility>
#include "send_heap.h"
#include "send_packet.h"
#include "send_writer.h"
#include "common_logging.h"

namespace spead
{
namespace send
{

template<typename Derived>
class rate_writer : public writer
{
private:
    struct state
    {
        packet_generator gen;
        packet cur_packet;
        countdown_handler &handler;

        state(const heap &h, countdown_handler &handler)
            : gen(h), handler(handler) {}
    };

protected:
    struct packet_callback
    {
        rate_writer<Derived> *self;
        std::unique_ptr<state> st;

        packet_callback(rate_writer *self, std::unique_ptr<state> &&st)
            : self(self), st(std::move(st))
        {
        }

        void operator()(const boost::system::error_code &ec, std::size_t bytes_transferred) const
        {
            if (ec)
            {
                // TODO: log the error
                st->countdown_handler();
            }
            else
            {
                self->next_packet(std::move(st));
            }
        }
    };

    void next_packet(std::unique_ptr<state> &&st)
    {
        st->cur_packet = st->gen.next_packet();
        if (st->cur_packet.buffers.empty())
        {
            st->countdown_handler();
        }
        else
        {
            static_cast<Derived *>(this)->async_send_packet(
                st->cur_packet, packet_callback(this, std::move(st)));
        }
    }

public:
    virtual void async_send_heap(const heap &h, write_handelr &handler) override
    {
        std::unique_ptr<state> st(new state(h, handler));
        next_packet(st);
    }
};

} // namespace send
} // namespace spead

#endif // SPEAD_SEND_RATE_WRITER_H
