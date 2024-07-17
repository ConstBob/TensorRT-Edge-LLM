#include "softmax_impl.h"

void run_softmax_fp16(void *dst,
                      const void *src,
                      const void *mask,
                      int s_inner,
                      int s_outer,
                      int b,
                      int h,
                      int warps_n,
                      bool has_alibi) {
    run_softmax<uint16_t, uint16_t>(
        dst, src, mask, s_inner, s_outer, b, h, 0.f, 0.f, warps_n, has_alibi);
}
