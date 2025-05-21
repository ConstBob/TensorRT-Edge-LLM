#pragma once
#include "common/common.h"
#include <cstdint>

struct EagleCommonParams
{
    int32_t batchSize{0};
    int32_t maxDecodingTokens{0};
    int32_t maxPathLen{0};
    int32_t hiddenDim{0};
    int32_t targetHiddenDim{0};
    int32_t topK{0};
    int32_t numLayers{0};
    int32_t numHead{0};
    int32_t hiddenSizePerHead{0};
    int32_t maxDraftTokens{0};
    int32_t maxSeqLen{0};
    int32_t kvCacheSeqLen{0};
    cudaStream_t stream;
};
struct AcceptDraftTokensByIdsWithPathsParams
{
    int64_t* outputIds{nullptr};
    int64_t* inputIdsDraftDecode{nullptr};
    //! input buffer [bs, maxDraftTokens]
    int64_t* draftIds{nullptr};
    //! input buffer [bs, maxDecodingTokens-1]
    int64_t* targetIds{nullptr};
    //! input buffer [bs]
    int32_t* contextLengths{nullptr};
    //! output buffer [bs]
    int64_t* acceptedLengths{nullptr};
    //! output buffer [bs]
    int32_t* bestPathIds{nullptr};
    //! output buffer [bs]
    int64_t* finishedFinal{nullptr};
    //! input buffer [bs,maxDecodingTokens,maxPathLen+1]
    int32_t* paths{nullptr};
    //! input buffer [bs]
    int64_t endIds{0};

    int32_t curTokensPerStep{1};

};

template <typename T>
struct UpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScoresParams
{
    //! input buffer [bs,topk,topk]
    int64_t* outputIdsAllDraft{nullptr};
    //! input buffer[bs,topk,hidden_dim]
    T* inputHiddenStatesDraft{nullptr};
    //! input buffer[bs,topk,topk]
    bool* treeMaskInit{nullptr};
    //! input buffer[bs,topk,topk*(depth+1)] depth=5
    bool* treeMaskInput{nullptr};
    //! input buffer[bs,topk,topk*(depth+1)] depth=5
    bool* treeMaskUpdate{nullptr};
    //! input buffer[bs,topk*(depth+1),topk*(depth+1)] depth=5
    bool* treeMaskUpdateforAttention{nullptr};

    //! input buffer[bs,topk]
    int64_t* treeIndices{nullptr};
    //! input buffer[bs]
    int32_t* curContextLengths{nullptr};

    //! input buffer[bs,topk*topk]
    float* cumScoresForThirdTopk{nullptr};
    //! input buffer[bs,topk]
    int64_t* outputIdsForThirdTopk{nullptr};
    //! input buffer[bs,maxPathLen*topk*topk]
    int64_t* allTokens{nullptr};
    //! input buffer[bs,topk]
    int64_t* draftVoc{nullptr};

    //! output buffer[bs,topk]--->[bs,maxPathLen,topk]
    int64_t* selectedOutputIdsDraft{nullptr};
    //! output buffer[bs,topk,hidden_dim]
    T* outputHiddenStatesDraft{nullptr};
    //! output buffer[bs,topk*(depth+1)]
    int32_t* treePositionIds{nullptr};

    //! output buffer[bs,topk*(depth+1),ceil(topk*(depth+1)/32)]
    int32_t* packedTreeMaskUpdateforAttention{nullptr};
    //! output buffer[bs,topk*(depth+1),ceil(topk*(depth+1)/32)]---valid shape=[bs*actualTokenLength*actualTokenLength]
    int32_t* packedTreeMaskUpdateforAttentionNoPadding{nullptr};

    //! input buffer[bs,topk]
    float* intermediateScores{nullptr};

    int32_t layerIdx{0};
    // topk*(depth+1) depth=pytorch code depth
    int32_t maxLength{0};
};

struct UpdateCumScoresAndParentsIdsParams
{
    //! input buffer [bs,topk,topk]
    float* outputLogProbsAllDraft{nullptr};
    //! input buffer [bs,topk]
    float* intermediateScores{nullptr};
    //! input buffer [bs,topk]
    int64_t* outputIdsCurrentDraft{nullptr};
    //! output buffer [bs,topk,topk]
    int64_t* parantsIds{nullptr};
    //! output buffer [bs,topk]
    float* cumScores{nullptr};
    int32_t layerIdx{0};

    int64_t bias{0};
};

struct AssembleDraftIdsAndPathAndMaskAndPositionIdsParams
{
    //! input buffer [bs,topk]
    int64_t* fourthTopKIds{nullptr};
    //! input buffer [bs,p_len] p_len=510
    int64_t* allDraftIds{nullptr};
    //! input buffer [bs,p_len]
    int64_t* allDraftIdsAncestors{nullptr};
    //! input buffer [bs,maxSeqLen]
    int64_t* modelInputIds{nullptr};
    //! input buffer [bs]
    int32_t* contextLengths{nullptr};
    //! output buffer [bs,maxDecodingTokens,maxDecodingTokens]
    bool* treeMask{nullptr};
    //! output buffer [bs,maxDecodingTokens]
    int32_t* positionIds{nullptr};
    //! output buffer [bs,maxDecodingDraftTokens]
    int64_t* draftIds{nullptr};
    //! output buffer [bs,maxDecodingDraftTokens]
    int64_t* draftIdsAncestors{nullptr};
    //! output buffer [bs,maxDecodingTokens,maxPathLen]
    int32_t* paths{nullptr};
    //! output buffer [bs]
    int32_t* validPathNum{nullptr};
    //! output buffer [bs,maxDecodingTokens,ceil(maxDecodingTokens/32)]
    int32_t* packedTreeMaskVerification{nullptr};
};

template <typename T>
struct UpdateKVCacheParams
{
    //! input buffer [numLayers, bs,2,numHead,maxSeqLen,hiddenSizePerHead]
    T* KVCache{nullptr};
    //! input buffer [bs,maxDecodingTokens,maxPathLen+1]
    int32_t* paths{nullptr};
    //! input buffer [bs]
    int32_t* bestPathIds{nullptr};
    //! input buffer [bs]
    int64_t* acceptedLengths{nullptr};
    //! input buffer [bs]
    int32_t* contextLengths{nullptr};
    //! input buffer [bs, maxDecodingTokens, hiddenDim]
    T* hiddenStates{nullptr};
    //! input buffer [bs, maxDecodingTokens, hiddenDim]
    T* hiddenStatesInputs{nullptr};
    //! input buffer [bs, maxDecodingTokens]
    int32_t* treePositionIds{nullptr};
    int32_t maxSeqLen{0};
};

struct InitCausalAttentionMaskParams
{
    bool* mask{nullptr};
    int32_t* packedMask{nullptr};
};

void dispatchUpdateCumScoresAndParentsIds(UpdateCumScoresAndParentsIdsParams const& params,EagleCommonParams const& commonParams);

void dispatchAssembleDraftIdsAndPathAndMaskAndPositionIds(
    AssembleDraftIdsAndPathAndMaskAndPositionIdsParams const& params, EagleCommonParams const& commonParams);

template <typename T>
void dispatchUpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScores(
    UpdateDraftInputIdsAndHiddenStatesAndTreeMaskAndPositionIdsAndInterScoresParams<T> const& params, EagleCommonParams const& commonParams);

void dispatchAcceptDraftTokensByIdsWithPaths(AcceptDraftTokensByIdsWithPathsParams const& params, EagleCommonParams const& commonParams);

template <typename T>
void dispatchUpdateKVCacheAndHiddenStatesAndTreePositionIds(UpdateKVCacheParams<T> const& params, EagleCommonParams const& commonParams);

void dispatchInitializeAttentionMaskCausal(InitCausalAttentionMaskParams const& params, EagleCommonParams const& commonParams);