#include <gtest/gtest.h>

#include "common/json.h"

// Simplest test with one level of Json object.
TEST(JsonParsingTest, basicOneLayer)
{
    drivellm::JsonRoot root;

    std::string const json = R"({"key0": "value0", "key1": 2, "key2": 3.14, "key3": true})";

    EXPECT_TRUE(root.parse(json));
    auto rootNode = root.getRoot();
    EXPECT_TRUE(rootNode.isObject());
    EXPECT_EQ(rootNode.size(), 4);

    auto childNode0 = rootNode["key0"];
    EXPECT_TRUE(childNode0.isString());
    EXPECT_EQ(childNode0.getString(), "value0");

    auto childNode1 = rootNode["key1"];
    EXPECT_TRUE(childNode1.isInteger());
    EXPECT_EQ(childNode1.getInteger(), 2);

    EXPECT_TRUE(rootNode.hasMember("key2"));
    auto childNode2 = rootNode[2];
    EXPECT_TRUE(childNode2.isFloat());
    EXPECT_EQ(childNode2.getFloat(), 3.14F);

    EXPECT_TRUE(rootNode.hasMember("key3"));
    auto childNode3 = rootNode[3];
    EXPECT_EQ(childNode3.getName(), "key3");
    EXPECT_TRUE(childNode3.isBool());
    EXPECT_EQ(childNode3.getBool(), true);
}

TEST(JsonParsingTest, QwenVLConfig)
{
    drivellm::JsonRoot root;
    // clang-format off
    // Selected Json config from Qwen2.5VL-3B
    std::string const json = R"(
{
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "vision_start_token_id": 151652,
  "vision_end_token_id": 151653,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "use_sliding_window": false,
  "vision_config": {
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 2048,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "window_size": 112,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "tokens_per_second": 2,
    "temporal_patch_size": 2
  },
  "rope_scaling": {
    "type": "mrope",
    "mrope_section": [
      16,
      24,
      24
    ]
  },
  "vocab_size": 151936
})";
    // clang-format on
    EXPECT_TRUE(root.parse(json));
    auto rootNode = root.getRoot();
    EXPECT_TRUE(rootNode.isObject());

    auto hiddenSizeNode = rootNode["hidden_size"];
    EXPECT_TRUE(hiddenSizeNode.isInteger());
    EXPECT_EQ(hiddenSizeNode.getInteger(), 2048);

    auto hiddenActNode = rootNode["hidden_act"];
    EXPECT_TRUE(hiddenActNode.isString());
    EXPECT_EQ(hiddenActNode.getString(), "silu");

    auto numKVHeadsNode = rootNode["num_key_value_heads"];
    EXPECT_TRUE(numKVHeadsNode.isInteger());
    EXPECT_EQ(numKVHeadsNode.getInteger(), 2);

    auto rmsNormEpsNode = rootNode["rms_norm_eps"];
    EXPECT_TRUE(rmsNormEpsNode.isFloat());
    EXPECT_EQ(rmsNormEpsNode.getFloat(), 1e-6F);
    EXPECT_FALSE(rmsNormEpsNode.isInteger());

    auto useSlidingWindowNode = rootNode["use_sliding_window"];
    EXPECT_TRUE(useSlidingWindowNode.isBool());
    EXPECT_EQ(useSlidingWindowNode.getBool(), false);

    auto visionConfigNode = rootNode["vision_config"];
    EXPECT_TRUE(visionConfigNode.isObject());
    {
        auto depthNode = visionConfigNode["depth"];
        EXPECT_TRUE(depthNode.isInteger());
        EXPECT_EQ(depthNode.getInteger(), 32);

        auto fullattBlockIndexesNode = visionConfigNode["fullatt_block_indexes"];
        EXPECT_TRUE(fullattBlockIndexesNode.isArray());
        EXPECT_EQ(fullattBlockIndexesNode.size(), 4);
        EXPECT_EQ(fullattBlockIndexesNode[0].getInteger(), 7);
        EXPECT_EQ(fullattBlockIndexesNode[1].getInteger(), 15);
        EXPECT_EQ(fullattBlockIndexesNode[2].getInteger(), 23);
        EXPECT_EQ(fullattBlockIndexesNode[3].getInteger(), 31);
    }

    auto ropeScalingNode = rootNode["rope_scaling"];
    EXPECT_TRUE(ropeScalingNode.isObject());
    {
        auto typeNode = ropeScalingNode["type"];
        EXPECT_TRUE(typeNode.isString());
        EXPECT_EQ(typeNode.getString(), "mrope");

        auto mropeSectionNode = ropeScalingNode["mrope_section"];
        EXPECT_TRUE(mropeSectionNode.isArray());
        EXPECT_EQ(mropeSectionNode.size(), 3);
        EXPECT_EQ(mropeSectionNode[0].getInteger(), 16);
        EXPECT_EQ(mropeSectionNode[1].getInteger(), 24);
        EXPECT_EQ(mropeSectionNode[2].getInteger(), 24);
    }
}

TEST(JsonParsingTest, SafeTensorHeader)
{
    // clang-format off
    std::string jsonHeader = R"(
{
  "__metadata__": {
    "format": "pytorch_v2.6",
    "library": "transformers",
    "model_type": "example_encoder",
    "notes": "Example header for demonstration"
  },
  "model.embedding.word_embeddings.weight": {
    "dtype": "F16",
    "shape": [32000, 4096],
    "data_offsets": [0, 262144000]
  },
  "model.encoder.layers.0.self_attn.q_proj.weight": {
    "dtype": "F16",
    "shape": [4096, 4096],
    "data_offsets": [262144000, 295567360]
  },
  "model.encoder.layers.0.self_attn.k_proj.bias": {
    "dtype": "F32",
    "shape": [4096],
    "data_offsets": [295567360, 295583744]
  },
  "model.encoder.layers.0.mlp.fc1.weight": {
    "dtype": "BF16",
    "shape": [11008, 4096],
    "data_offsets": [295583744, 385777664]
  },
  "lm_head.weight": {
    "dtype": "F32",
    "shape": [32000, 4096],
    "data_offsets": [385777664, 910966784]
  }
})";
    // clang-format on
    drivellm::JsonRoot root;
    EXPECT_TRUE(root.parse(jsonHeader));
    auto rootNode = root.getRoot();
    EXPECT_TRUE(rootNode.isObject());

    EXPECT_EQ(rootNode.size(), 6);

    auto metadataNode = rootNode["__metadata__"];
    EXPECT_TRUE(metadataNode.isObject());
    {
        auto formatNode = metadataNode["format"];
        EXPECT_TRUE(formatNode.isString());
        EXPECT_EQ(formatNode.getString(), "pytorch_v2.6");

        auto libraryNode = metadataNode["library"];
        EXPECT_TRUE(libraryNode.isString());
        EXPECT_EQ(libraryNode.getString(), "transformers");
    }

    auto weightNode1 = rootNode[1];
    EXPECT_TRUE(weightNode1.isObject());
    {
        auto dtypeNode = weightNode1["dtype"];
        EXPECT_TRUE(dtypeNode.isString());
        EXPECT_EQ(dtypeNode.getString(), "F16");

        auto shapeNode = weightNode1["shape"];
        EXPECT_TRUE(shapeNode.isArray());
        EXPECT_EQ(shapeNode.size(), 2);
        EXPECT_EQ(shapeNode[0].getInteger(), 32000);
        EXPECT_EQ(shapeNode[1].getInteger(), 4096);

        auto dataOffsetsNode = weightNode1["data_offsets"];
        EXPECT_TRUE(dataOffsetsNode.isArray());
        EXPECT_EQ(dataOffsetsNode.size(), 2);
        EXPECT_EQ(dataOffsetsNode[0].getInteger(), 0);
        EXPECT_EQ(dataOffsetsNode[1].getInteger(), 262144000);
    }

    auto weightNode4 = rootNode[4];
    EXPECT_TRUE(weightNode4.isObject());
    {
        auto dtypeNode = weightNode4["dtype"];
        EXPECT_TRUE(dtypeNode.isString());
        EXPECT_EQ(dtypeNode.getString(), "BF16");

        auto shapeNode = weightNode4["shape"];
        EXPECT_TRUE(shapeNode.isArray());
        EXPECT_EQ(shapeNode.size(), 2);
        EXPECT_EQ(shapeNode[0].getInteger(), 11008);
        EXPECT_EQ(shapeNode[1].getInteger(), 4096);

        auto dataOffsetsNode = weightNode4["data_offsets"];
        EXPECT_TRUE(dataOffsetsNode.isArray());
        EXPECT_EQ(dataOffsetsNode.size(), 2);
        EXPECT_EQ(dataOffsetsNode[0].getInteger(), 295583744);
        EXPECT_EQ(dataOffsetsNode[1].getInteger(), 385777664);
    }
}