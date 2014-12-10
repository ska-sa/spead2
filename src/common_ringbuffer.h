#ifndef SPEAD_COMMON_RINGBUFFER_H
#define SPEAD_COMMON_RINGBUFFER_H

#include <mutex>
#include <condition_variable>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <cassert>

namespace spead
{

class ringbuffer_full : public std::runtime_error
{
public:
    ringbuffer_full() : std::runtime_error("ring buffer is full") {}
};

class ringbuffer_empty : public std::runtime_error
{
public:
    ringbuffer_empty() : std::runtime_error("ring buffer is empty") {}
};

/**
 * Thread-safe ring buffer with blocking and non-blocking pop, but only
 * non-blocking push. It supports non-copyable objects using move semantics.
 */
template<typename T>
class ringbuffer
{
private:
    typedef typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_type;

    std::unique_ptr<storage_type[]> storage;
    const std::size_t cap;
    std::size_t head = 0;
    std::size_t tail = 0;
    std::size_t len = 0;

    T *get(std::size_t idx);

    std::size_t next(std::size_t idx)
    {
        idx++;
        if (idx == cap)
            idx = 0;
        return idx;
    }

protected:
    // This is all protected rather than private to allow alternative
    // blocking strategies
    std::mutex mutex;
    std::condition_variable data_cond; // signalled when data is added

    bool empty_unlocked() const { return len == 0; }
    // Assumes there is data (caller must check)
    T pop_unlocked();

public:
    explicit ringbuffer(std::size_t cap);
    ~ringbuffer();

    // Throws ringbuffer_full if necessary
    void try_push(T &&value);

    // Throws ringbuffer_full if necessary
    template<typename... Args>
    void try_emplace(Args&&... args);

    // Throws ringbuffer_empty if necessary
    T try_pop();

    // Blocks if necessary
    T pop();
};

template<typename T>
ringbuffer<T>::ringbuffer(std::size_t cap)
    : storage(new storage_type[cap]), cap(cap)
{
    assert(cap > 0);
}

template<typename T>
ringbuffer<T>::~ringbuffer()
{
    while (len > 0)
    {
        get(head)->~T();
        head = next(head);
        len--;
    }
}

template<typename T>
T *ringbuffer<T>::get(std::size_t idx)
{
    return static_cast<T*>(static_cast<void *>(&storage[idx]));
}

template<typename T>
void ringbuffer<T>::try_push(T &&value)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (len == cap)
        throw ringbuffer_full();
    new (get(tail)) T(std::move(value));
    tail = next(tail);
    len++;
    lock.unlock();
    data_cond.notify_one();
}

template<typename T>
template<typename... Args>
void ringbuffer<T>::try_emplace(Args&&... args)
{
    std::unique_lock<std::mutex> lock(mutex);
    if (len == cap)
        throw ringbuffer_full();
    new (get(tail)) T(std::forward<Args>(args)...);
    tail = next(tail);
    len++;
    lock.unlock();
    data_cond.notify_one();
}

template<typename T>
T ringbuffer<T>::pop_unlocked()
{
    T result = std::move(*get(head));
    head = next(head);
    len--;
    return result;
}

template<typename T>
T ringbuffer<T>::try_pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    if (len == 0)
        throw ringbuffer_empty();
    return pop_unlocked();
}

template<typename T>
T ringbuffer<T>::pop()
{
    std::unique_lock<std::mutex> lock(mutex);
    while (len == 0)
        data_cond.wait(lock);
    return pop_unlocked();
}

} // namespace spead

#endif // SPEAD_COMMON_RINGBUFFER_H
