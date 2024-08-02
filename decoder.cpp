#include <decoder.h>
#include <cassert>
using namespace nvinfer1;
using namespace std;

bool Decoder::setup(std::filesystem::path& const fp, cudaStream_t& const stream){
    try{
        mStream = stream;
        mLogger = std::make_shared<Logger>();
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        StreamReader* _sr = new StreamReader(fp);
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(_sr));
        mContextExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        mGenerationExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        assert(mEngine->getNbOptimizationProfiles() == 2 && "The engine requires 2 optimization profiles");
        mContextExecutionContext->setOptimizationProfileAsync(0, stream_);
        mGenerationExecutionContext->setOptimizationProfileAsync(1, stream_);
        isSetup = true;
    }
    catch (std::exception const& e)
    {
        isSetup = false;
        std::cerr << e.what() << std::endl;
        return false;
    }
    if (!validateAndFillConfig()){
        return false;
    }
    return true;
}

// Helper function to check 2 dims are equal.
bool checkDimsEqual(Dims& const A, Dims& const B){
    if (A.nbDims != B.nbDims){
        return false;
    }
    for (int32_t i = 0; i < A.nbDims; ++i){
        if (A.d[i] != B.d[i]){
            return false;
        }
    }
    return true;
}

// Helper function to check a certain input tensor has static shape
bool Decoder::checkStaticShape(const string& name){
    for (int32_t i = 0; i < mEngine->getNbOptimizationProfiles(); ++i){
        Dims minShape = mEngine->getProfileShape(keyName, i, OptProfileSelector::kMIN);
        Dims optShape = mEngine->getProfileShape(keyName, i, OptProfileSelector::kOPT);
        Dims maxShape = mEngine->getProfileShape(keyName, i, OptProfileSelector::kMAX);
        if (!checkDimsEqual(minShape, optShape)){
            return false;
        }
        if (!checkDimsEqual(optShape, maxShape)){
            return false;
        }
    }
    return true;
}

bool Decoder::validateAndFillConfig(){
    assert(isSetup && "Please make sure the decoder has been set up.");
    int64_t batchSize;
    int64_t numHead;
    int64_t hiddenSizePerHead;
    int64_t maxInputLength;
    int64_t maxLength;
    int64_t nbIOs = static_cast<int64_t>(mEngine-> getNbIOTensors());
    assert(nbIOs % 2 == 0, "Number of IO Tensor needs to be a multiple of 4.")
    int64_t numLayers = (nvIOs - 4) // 4;
    // Check input_ids, attention_mask and position_ids
    std::string inputIdsName = "input_ids";
    assert(checkStaticShape(inputIdsName) && fmtstr("%s should be static", inputIdsName));
    Dims inputIdsShapeContext = mEngine->getProfileShape(inputIdsName, 0, OptProfileSelector::kMIN);
    batchSize = inputIdsShapeContext.d[0];
    maxInputLength = inputIdsShapeContext.d[1];
    Dims inputIdsShapeGeneration = mEngine->getProfileShape(inputIdsName, 1, OptProfileSelector::kMIN);
    assert(inputIdsShapeGeneration.d[0] == batchSize && inputIdsShapeGeneration.d[1] == 1);

    std::string attentionMaskName = "attention_mask";
    assert(checkStaticShape(attentionMaskName) && fmtstr("%s should be static", attentionMaskName));
    Dims attentionMaskShapeContext = mEngine->getProfileShape(attentionMaskName, 0, OptProfileSelector::kMIN);
    assert(batchSize == attentionMaskShapeContext.d[0] && maxInputLength == attentionMaskShapeContext.d[1]);
    Dims attentionMaskShapeGeneration = mEngine->getProfileShape(attentionMaskName, 1, OptProfileSelector::kMIN);
    assert(batchSize == attentionMaskShapeGeneration.d[0]);
    maxLength = attentionMaskShapeGeneration.d[1];

    std::string positionIdsName = "positon_ids";
    assert(checkStaticShape(positionIdsName) && fmtstr("%s should be static", positionIdsName));
    Dims positionIdsShapeContext = mEngine->getProfileShape(positionIdsName, 0, OptProfileSelector::kMIN);
    assert(batchSize == positionIdsShapeContext.d[0] && maxInputLength == positionIdsShapeContext.d[1]);
    Dims positionIdsShapeGeneration = mEngine->getProfileShape(positionIdsName, 1, OptProfileSelector::kMIN);
    assert(batchSize == positionIdsShapeGeneration.d[0] && positionIdsShapeGeneration.d[1] == 1);

    for (int32_t i = 0; i < numLayers; ++i){
        std::string keyName = fmtstr("past_key_values.%d.key", i);
        std::string valueName = fmtstr("past_key_values.%d.value", i);
        assert(checkStaticShape(keyName) && fmtstr("%s should be static", keyName));
        assert(checkStaticShape(valueName) && fmtstr("%s should be static", valueName));
        Dims keyShapeContext = mEngine->getProfileShape(keyName, 0, OptProfileSelector::kMIN);
        Dims keyShapeGeneration = mEngine->getProfileShape(keyName, 1, OptProfileSelector::kMIN);
        Dims valueShapeContext = mEngine->getProfileShape(valueName, 0, OptProfileSelector::kMIN);
        Dims valueShapeGeneration = mEngine->getProfileShape(valueName, 1, OptProfileSelector::kMIN);
        assert(checkDimsEqual(keyShapeContext, valueShapeContext));
        assert(checkDimsEqual(keyShapeGeneration, valueShapeGeneration));
        assert(keyShapeContext.nbDims == 4 && keyShapeGeneration.nbDims == 4, "key value pairs should have 4 dimensions");
        if (i == 0){
            assert(keyShapeContext.d[0] == batchSize);
            numHead = keyShapeContext.d[1];
            assert(keyShapeContext.d[2] == 0);
            hiddenSizePerHead = keyShapeContext.d[3];
        }
        else{
            assert((keyShapeContext.d[0] == batchSize) && (keyShapeContext.d[1] == numHead) && (keyShapeContext.d[2] == 0) && (keyShapeContext.d[3] == hiddenSizePerHead));
        }
        assert((keyShapeGeneration.d[0] == batchSize) && (keyShapeGeneration.d[1] == numHead) && (keyShapeGeneration.d[2] == (maxLength - 1)) && (keyShapeGeneration.d[3] == hiddenSizePerHead));
    }

    mConfig = {batchSize, numHead, hiddenSizePerHead, maxInputLength, maxLength, numLayers};

    return 0;
}

bool Decoder::allocateBuffer(){
    if (mConfig.batchSize == 0){
        validateAndFillConfig();
    }

}
