namespace tensorrt_llm
{
namespace kernels
{

extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_8_m_8_sm_86_cubin[];
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_8_m_8_sm_86_cubin_len;

enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32,
    DATA_TYPE_BF16,
    DATA_TYPE_E4M3,
    DATA_TYPE_E5M2
};

constexpr int32_t kSM_80 = 80;
constexpr int32_t kSM_86 = 86;
constexpr int32_t kSM_89 = 89;
constexpr int32_t kSM_90 = 90;

static const struct XQAKernelMetaInfo
{
    Data_type mDataType;
    Data_type mKVDataType;
    unsigned int mHeadDim;
    unsigned int mBeamWidth;
    unsigned int mNumQHeadsOverKV;
    unsigned int mMTileSize;
    unsigned int mTokensPerPage;
    bool mPagedKVCache;
    bool mMultiQueryTokens;
    unsigned int mSM;
    const unsigned long long* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
} sXqaKernelMetaInfo[] = {
{DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, 8, 0, false, false, kSM_86, xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_8_m_8_sm_86_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_nqpkv_8_m_8_sm_86_cubin_len, "kernel_mha"} 
};

} // namespace kernels
} // namespace tensorrt_llm