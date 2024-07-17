#include "softmax_impl.h"

void run_softmax_e4m3(void *dst,
                      const void *src,
                      const void *mask,
                      int s_inner,
                      int s_outer,
                      int b,
                      int h,
                      float scale_softmax,
                      int warps_n,
                      bool has_alibi) {
    run_softmax<fmha::e4m3_t, float>(
        dst, src, mask, s_inner, s_outer, b, h, 0.f, scale_softmax, warps_n, has_alibi);
}
