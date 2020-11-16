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
 */

#include <gdrapi.h>
#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <spead2/recv_stream.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_udp.h>
#include <spead2/common_ringbuffer.h>
#include <spead2/common_memory_allocator.h>
#include <spead2/common_memory_pool.h>

const char *sum_src = R"(
extern "C" __global__ void sum(const unsigned int *data, unsigned long long *out, int n)
{
    unsigned long long ans = 0;
    for (int i = 0; i < n; i++)
        ans += data[i];
    *out = ans;
}
)";


#define CUDA_CHECK(cmd)             \
    ({                              \
        CUresult result = (cmd);    \
        if (result != CUDA_SUCCESS) \
        {                           \
            std::cerr << "CUDA error: " << #cmd << "\n"; \
            std::exit(1);           \
        }                           \
        result;                     \
    })

#define NVRTC_CHECK(cmd)             \
    ({                              \
        nvrtcResult result = (cmd);    \
        if (result != NVRTC_SUCCESS) \
        {                           \
            std::cerr << "nvrtc error: " << #cmd << "\n"; \
            std::exit(1);           \
        }                           \
        result;                     \
    })

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

// Compile CUDA source to PTX
static std::vector<char> compile(const char *src)
{
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src, NULL, 0, NULL, NULL));
    NVRTC_CHECK(nvrtcCompileProgram(prog, 0, NULL));
    std::size_t size;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &size));
    std::vector<char> out(size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, out.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    return out;
}

class gdrapi_memory_allocator : public spead2::memory_allocator
{
private:
    struct metadata
    {
        gdr_mh_t mh;
        CUdeviceptr base;          // address to free
        CUdeviceptr dptr;          // device address corresponding to the pointer
        std::size_t padded_size;   // size for unmapping
    };

    gdr_t gdr;

    virtual void free(std::uint8_t *ptr, void *user) override;

public:
    gdrapi_memory_allocator();
    ~gdrapi_memory_allocator();

    virtual pointer allocate(std::size_t size, void *hint) override;

    static CUdeviceptr get_device_ptr(const pointer &ptr);
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
    meta->padded_size = (size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    // We have to allocate more than requested so that we can manually align
    // to the page size. See https://github.com/NVIDIA/gdrcopy/issues/52
    std::size_t alloc_size = meta->padded_size + GPU_PAGE_SIZE - 1;
    CUDA_CHECK(cuMemAlloc(&meta->base, alloc_size));
    // Round up to the next GPU page
    meta->dptr = (meta->base + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    INT_CHECK(gdr_pin_buffer(gdr, meta->dptr, meta->padded_size, 0, 0, &meta->mh));
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
    CUDA_CHECK(cuMemFree(meta->base));
    delete meta;
}

CUdeviceptr gdrapi_memory_allocator::get_device_ptr(const pointer &ptr)
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

int main()
{
    CUdevice device;
    CUcontext ctx;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr dsum;

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuCtxCreate(&ctx, 0, device));
    std::vector<char> ptx = compile(sum_src);
    CUDA_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "sum"));
    CUDA_CHECK(cuMemAlloc(&dsum, sizeof(unsigned long long)));

    auto raw_alloc = std::make_shared<gdrapi_memory_allocator>();
    auto alloc = std::make_shared<spead2::memory_pool>(0, 32 * 1024 * 1024, 8, 8, raw_alloc);

    spead2::thread_pool tp;
    spead2::recv::stream_config config;
    config.set_memory_allocator(alloc);
    config.set_memcpy(spead2::MEMCPY_NONTEMPORAL);
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
                if (item.id == 0x1000 && !item.is_immediate)
                {
                    const gdrapi_memory_allocator::pointer &ptr = heap.get_payload();
                    std::size_t offset = item.ptr - ptr.get();
                    CUdeviceptr dptr = raw_alloc->get_device_ptr(ptr) + offset;
                    int n = item.length / sizeof(std::uint32_t);
                    void *args[] = {&dptr, &dsum, &n};
                    CUDA_CHECK(cuLaunchKernel(
                        kernel,
                        1, 1, 1,
                        1, 1, 1,
                        0, NULL,
                        args, NULL));
                    unsigned long long hsum = 0;
                    CUDA_CHECK(cuMemcpy((CUdeviceptr) &hsum, dsum, sizeof(hsum)));
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
