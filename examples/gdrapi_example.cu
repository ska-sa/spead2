/* Copyright 2020 National Research Foundation (SARAO)
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* This examples shows how one can use gdrcopy (with an NVIDIA data
 * centre GPU) to reduce the number of system memory copies when
 * receiving data.
 *
 * Packets arriving from the network are still stored in system memory,
 * but heaps are assembled directly into GPU memory.
 *
 * To run the demo, run this program, then separately (on the same machine)
 * run "spead2_send --heaps 20 localhost:8888 --verify" (any heap count will
 * do). This program will print out the average of the random numbers that
 * appear in each heap.
 *
 * It should be noted that this demo is not particularly optimal. It uses the
 * kernel's UDP stack (which will involve copies), there is no overlap between
 * GPU transfers and computation, the GPU computation is inefficient etc. It
 * is intended as an example of using custom memory allocators, rather than
 * a ready-to-use code.
 */

#include <gdrapi.h>
#include <cuda.h>
#include <iostream>
#include <spead2/recv_stream.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_udp.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_memory_pool.h>

/**
 * Add up the @a n items in @a data, writing the result to @a out. This should
 * be run with a single CUDA thread and block.
 */
__global__ void sum(const unsigned int *data, unsigned long long *out, int n)
{
    unsigned long long ans = 0;
    for (int i = 0; i < n; i++)
        ans += data[i];
    *out = ans;
}

/// Crash out if a CUDA call fails
#define CUDA_CHECK(cmd)             \
    ({                              \
        cudaError_t result = (cmd); \
        if (result != cudaSuccess)  \
        {                           \
            std::cerr << "CUDA error: " << #cmd << "\n"; \
            std::exit(1);           \
        }                           \
        result;                     \
    })

/// Crash out if a call returning an integer does not return 0
#define INT_CHECK(cmd)              \
    ({                              \
        int result = (cmd);         \
        if (result != 0)            \
        {                           \
            std::cerr << "Error: " << #cmd << "\n"; \
            std::exit(1);           \
        }                           \
        result;                     \
    })

/**
 * Custom memory allocator that allocates GPU memory. The memory is mapped to
 * the CPU's address space using gdrcopy (aka gdrapi).
 */
class gdrapi_memory_allocator : public spead2::memory_allocator
{
private:
    /// Information stored alongside each memory allocation.
    struct metadata
    {
        gdr_mh_t mh;               ///< memory handle for interacting with gdrapi
        void *base;                ///< device address to free
        void *dptr;                ///< device address corresponding to the pointer
        std::size_t padded_size;   ///< size for unmapping
    };

    gdr_t gdr;                     ///< Global handle for interacting with gdrapi

    virtual void free(std::uint8_t *ptr, void *user) override;

public:
    gdrapi_memory_allocator();
    ~gdrapi_memory_allocator();

    virtual pointer allocate(std::size_t size, void *hint) override;

    /**
     * Get the CUDA device pointer from a pointer allocated with this allocator.
     *
     * It is undefined behaviour to pass a pointer not obtained from this allocator.
     */
    static void *get_device_ptr(const pointer &ptr);

    /**
     * Get the gdrapi memory handle from a pointer allocated with this allocator.
     *
     * It is undefined behaviour to pass a pointer not obtained from this allocator.
     */
    static gdr_mh_t get_mh(const pointer &ptr);
};

gdrapi_memory_allocator::gdrapi_memory_allocator()
{
    gdr = gdr_open();
    if (!gdr)
    {
        std::cerr << "gdr_open failed (do you have gdrdrv loaded and a data centre GPU?)\n";
        std::exit(1);
    }
}

gdrapi_memory_allocator::~gdrapi_memory_allocator()
{
    INT_CHECK(gdr_close(gdr));
}

gdrapi_memory_allocator::pointer gdrapi_memory_allocator::allocate(std::size_t size, void *hint)
{
    std::unique_ptr<metadata> meta(new metadata);
    // Mapping is done in GPU page granularity, so round up size to a GPU page.
    meta->padded_size = (size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    // We have to allocate more than requested so that we can manually align
    // to the page size. See https://github.com/NVIDIA/gdrcopy/issues/52
    std::size_t alloc_size = meta->padded_size + GPU_PAGE_SIZE - 1;
    CUDA_CHECK(cudaMalloc(&meta->base, alloc_size));
    // Round up device pointer to the next GPU page boundary
    meta->dptr = reinterpret_cast<void *>(
        (reinterpret_cast<std::uintptr_t>(meta->base) + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK
    );
    // Pin and map the GPU memory
    INT_CHECK(gdr_pin_buffer(gdr, CUdeviceptr(meta->dptr), meta->padded_size, 0, 0, &meta->mh));
    void *hptr;
    INT_CHECK(gdr_map(gdr, meta->mh, &hptr, meta->padded_size));
    return pointer(
        static_cast<std::uint8_t *>(hptr),
        deleter(shared_from_this(), meta.release()));
}

void gdrapi_memory_allocator::free(std::uint8_t *ptr, void *user)
{
    metadata *meta = static_cast<metadata *>(user);
    INT_CHECK(gdr_unmap(gdr, meta->mh, ptr, meta->padded_size));
    INT_CHECK(gdr_unpin_buffer(gdr, meta->mh));
    CUDA_CHECK(cudaFree(meta->base));
    delete meta;
}

void *gdrapi_memory_allocator::get_device_ptr(const pointer &ptr)
{
    metadata *meta = static_cast<metadata *>(ptr.get_deleter().get_user());
    assert(meta != nullptr);
    return meta->dptr;
}

gdr_mh_t gdrapi_memory_allocator::get_mh(const pointer &ptr)
{
    metadata *meta = static_cast<metadata *>(ptr.get_deleter().get_user());
    assert(meta != nullptr);
    return meta->mh;
}

/**
 * Wrap gdr_copy_to_mapping for copies. One could also use
 * spead2::MEMCPY_NONTEMPORAL.
 */
static void custom_memcpy(const spead2::memory_allocator::pointer &allocation,
                          const spead2::recv::packet_header &packet)
{
    gdr_copy_to_mapping(
        gdrapi_memory_allocator::get_mh(allocation),
        allocation.get() + packet.payload_offset,
        packet.payload,
        packet.payload_length);
}

int main()
{
    // Storage for the result of each sum
    unsigned long long *dsum;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dsum), sizeof(unsigned long long)));

    auto raw_alloc = std::make_shared<gdrapi_memory_allocator>();
    // Create a memory pool so that we don't allocate and release GPU memory
    // with every new heap.
    auto alloc = std::make_shared<spead2::memory_pool>(0, 32 * 1024 * 1024, 8, 8, raw_alloc);

    // Set up the stream
    spead2::thread_pool tp;
    spead2::recv::stream_config config;
    config.set_memory_allocator(alloc);
    config.set_memcpy(custom_memcpy);
    spead2::recv::ring_stream<> stream(tp, config);
    stream.emplace_reader<spead2::recv::udp_reader>(
        boost::asio::ip::udp::endpoint(
            boost::asio::ip::address_v4(),
            8888));

    while (true)
    {
        try
        {
            spead2::recv::heap heap = stream.pop();
            for (const auto &item : heap.get_items())
            {
                // spead2_send sends the first item with ID 0x1000
                if (item.id == 0x1000 && !item.is_immediate)
                {
                    const gdrapi_memory_allocator::pointer &ptr = heap.get_payload();
                    /* The item might not be at the start of the payload (e.g.
                     * if there are descriptors. Determine the offset in the
                     * CPU mapping, and apply it to the GPU pointer.
                     */
                    std::size_t offset = item.ptr - ptr.get();
                    void *dptr = static_cast<std::uint8_t *>(raw_alloc->get_device_ptr(ptr)) + offset;
                    /* Sum up the elements, and print the average. This ignores
                     * the fact that the elements are big endian, but
                     * byte-swapped random numbers are still random numbers.
                     */
                    int n = item.length / sizeof(std::uint32_t);
                    sum<<<1, 1>>>(reinterpret_cast<std::uint32_t *>(dptr), dsum, n);
                    unsigned long long hsum = 0;
                    CUDA_CHECK(cudaMemcpy(&hsum, dsum, sizeof(hsum), cudaMemcpyDeviceToHost));
                    std::cout << std::fixed << double(hsum) / n << '\n';
                }
            }
        }
        catch (spead2::ringbuffer_stopped &)
        {
            break;
        }
    }
}
