#pragma once

#include "common.h"
#include "cuda_runtime_api.h"

class CudaEvent
{

public:
    using pointer = cudaEvent_t;

    explicit CudaEvent(unsigned int flags = cudaEventDefault)
    {
        pointer event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
        mEvent = EventPtr(event, Deleter());
    }

    pointer get() const
    {
        return mEvent.get();
    }

    void synchronize() const
    {
        CUDA_CHECK(cudaEventSynchronize(get()));
    }

    void record() const
    {
        CUDA_CHECK(cudaEventRecord(get()));
    }

private:
    class Deleter
    {
    public:
        Deleter() = default;

        constexpr void operator()(pointer event) const
        {
            if (event != nullptr)
            {
                CUDA_CHECK(cudaEventDestroy(event));
            }
        }
    };

    using element_type = std::remove_pointer_t<pointer>;
    using EventPtr = std::unique_ptr<element_type, Deleter>;

    EventPtr mEvent;
};