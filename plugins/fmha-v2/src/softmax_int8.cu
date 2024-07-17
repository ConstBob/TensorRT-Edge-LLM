#include "softmax_impl.h"

void run_softmax_int8(void *dst,
                      const void *src,
                      const void *mask,
                      int s_inner,
                      int s_outer,
                      int b,
                      int h,
                      float scale_bmm1,
                      float scale_softmax,
                      int warps_n,
                      bool has_alibi) {
    run_softmax<int8_t, int32_t>(
        dst, src, mask, s_inner, s_outer, b, h, scale_bmm1, scale_softmax, warps_n, has_alibi);
}
