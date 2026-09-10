/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Contract tests: host-side, no GPU and no engine. The statistics below are the
// pi05_libero checkpoint's own norm_stats.json.

#include "runtime/pi05Policy.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr char const* kLiberoNormStats = R"JSON({
  "norm_stats": {
    "state": {
      "q01": [-0.3524468903720379, -0.26824864755272865, 0.04083745917417109, 1.5317653684616088,
              -2.7152330031871794, -1.076538143157959, 0.001715825623134151, -0.04003722561979666],
      "q99": [0.13891278689503672, 0.3251991607129573, 1.2568962905768304, 3.26276856803894,
              2.4437233173847197, 0.5638469840288161, 0.04030780866963323, -0.0017131616945378486]
    },
    "actions": {
      "q01": [-0.747375, -0.796125, -0.9375, -0.11580300460159779, -0.16942972007393836,
              -0.194502209174633, -1.0],
      "q99": [0.937125, 0.8594999999999999, 0.937125, 0.1402260055720806, 0.18103543001413347,
              0.3115457148551941, 0.9996]
    }
  }
})JSON";

//! One export run's stamp; the engines staged beside a contract carry the same one.
constexpr char const* kExportId = "1111111111111111aaaaaaaaaaaaaaaa";

constexpr char const* kLiberoCameras
    = R"JSON({"present": ["observation.images.image", "observation.images.image2"], "empty": 1})JSON";

//! The pi05_libero contract as the exporter writes it.
std::string liberoContract()
{
    return std::string(R"JSON({
  "contract_version": 1,
  "model_family": "pi05",
  "policy_config": "pi05_libero",
  "discrete_state_input": false,
  "export_id": ")JSON")
        + kExportId + R"JSON(",
  "state": {"dim": 8, "max_dim": 32, "num_bins": 256, "eps": 1e-08},
  "action": {"dim": 7, "max_dim": 32, "horizon": 10},
  "cameras": )JSON"
        + kLiberoCameras + R"JSON(,
  "image_resolution": [224, 224],
  "tokenizer": {"max_length": 200}
})JSON";
}

//! Stage one engine directory's worth of contract files and hand back its path.
class Pi05PolicyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDir = std::filesystem::temp_directory_path() / "edgellm_pi05_policy_test";
        std::filesystem::remove_all(mDir);
        std::filesystem::create_directories(mDir / "assets");
        std::ofstream(mDir / "assets" / "norm_stats.json") << kLiberoNormStats;
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
    }

    pi05::Pi05Policy makePolicy()
    {
        std::ofstream(mDir / "policy.json") << liberoContract();
        return pi05::Pi05Policy(mDir.string());
    }

    std::filesystem::path mDir;
};

//! openpi resizes uint8 and only then maps to [-1, 1], so the goldens are the reference's
//! own bytes.
void expectMatchesGolden(std::vector<float> const& planar, std::vector<uint8_t> const& golden)
{
    ASSERT_EQ(planar.size(), golden.size());
    for (size_t i = 0; i < golden.size(); ++i)
    {
        EXPECT_NEAR(planar[i], static_cast<float>(golden[i]) / 255.0F * 2.0F - 1.0F, 1e-6F) << "element " << i;
    }
}

} // namespace

TEST_F(Pi05PolicyTest, QuantileActionsUnnormalizeToRobotUnits)
{
    pi05::Pi05Policy const policy = makePolicy();
    // One timestep of the 32-wide chunk the engine emits; only the first 7 are the robot's.
    std::vector<float> normalized(32, 7.0F);
    std::vector<float> const row{1.0F, 0.5F, 0.0F, -0.5F, -1.0F, 0.25F, -0.25F};
    std::copy(row.begin(), row.end(), normalized.begin());

    std::vector<float> const robot = policy.unnormalizeActions(normalized, 1, 32);

    ASSERT_EQ(robot.size(), 7U);
    // +-1 land exactly on q99 and q01, so dims 0 and 4 pin the endpoints of the map.
    std::vector<float> const expected{
        0.937124968F, 0.445593774F, -0.000187516F, -0.0517957509F, -0.169429719F, 0.121777773F, -0.250150025F};
    for (size_t d = 0; d < expected.size(); ++d)
    {
        EXPECT_NEAR(robot[d], expected[d], 1e-6F) << "dim " << d;
    }
}

//! The 5x3 case upscales one axis; only a real downscale exercises the widened filter
//! support, which is where a fixed two-tap bilinear diverges. 8x8 -> 7x7 is LIBERO's
//! 256 -> 224 ratio, so it separates the two at the smallest size that still does.
//! 8x8 -> 7x7 is LIBERO's 256 -> 224 ratio. Only a real downscale widens the filter
//! support, which is where a fixed two-tap bilinear diverges from the reference.
TEST_F(Pi05PolicyTest, DownscaleAtTheLiberoRatioMatchesTheReference)
{
    std::vector<uint8_t> const rgb{78, 204, 64, 34, 16, 250, 233, 32, 103, 122, 13, 204, 138, 172, 208, 127, 124, 100,
        12, 151, 222, 203, 251, 153, 211, 235, 80, 182, 176, 41, 88, 7, 141, 241, 73, 124, 223, 123, 222, 37, 84, 55,
        200, 102, 211, 60, 159, 237, 53, 138, 57, 140, 55, 21, 7, 18, 118, 67, 241, 138, 148, 221, 56, 33, 89, 208, 34,
        193, 76, 167, 197, 242, 177, 243, 188, 250, 92, 195, 51, 159, 144, 36, 75, 222, 86, 85, 118, 94, 59, 57, 79, 37,
        219, 116, 234, 130, 213, 138, 152, 113, 99, 19, 176, 169, 19, 47, 154, 254, 119, 163, 122, 70, 255, 89, 16, 219,
        40, 224, 81, 35, 126, 236, 5, 89, 126, 246, 188, 201, 215, 100, 100, 63, 81, 192, 156, 171, 194, 167, 128, 117,
        191, 124, 43, 131, 2, 183, 247, 240, 153, 163, 21, 209, 158, 222, 211, 214, 98, 50, 144, 140, 55, 227, 117, 251,
        0, 40, 29, 251, 223, 53, 151, 34, 104, 187, 90, 52, 42, 26, 241, 78, 231, 69, 193, 141, 27, 163, 232, 210, 0,
        212, 206, 123};
    std::vector<float> planar(3U * 7U * 7U);
    pi05::resizeWithPad(rgb.data(), 8, 8, 7, 7, planar.data());

    // openpi_client.image_tools.resize_with_pad on the same pixels.
    std::vector<uint8_t> const golden{94, 100, 183, 147, 127, 63, 156, 168, 141, 115, 198, 100, 121, 100, 115, 139, 36,
        90, 83, 53, 131, 193, 163, 86, 72, 90, 57, 150, 150, 121, 143, 88, 118, 98, 150, 64, 162, 114, 141, 127, 128,
        73, 166, 72, 71, 134, 167, 193, 188, 182, 38, 26, 94, 136, 136, 220, 197, 106, 53, 135, 110, 116, 162, 161, 71,
        113, 172, 112, 117, 198, 175, 115, 145, 120, 144, 136, 153, 184, 136, 162, 151, 203, 173, 74, 189, 163, 197,
        135, 92, 181, 63, 81, 103, 139, 151, 45, 165, 183, 91, 186, 141, 200, 138, 185, 176, 67, 63, 132, 152, 128, 153,
        228, 105, 100, 145, 125, 123, 137, 187, 160, 100, 142, 179, 92, 114, 91, 78, 152, 138, 140, 139, 133, 100, 181,
        183, 173, 160, 186, 97, 190, 170, 150, 77, 143, 178, 48, 120};
    expectMatchesGolden(planar, golden);
}

//! 5x3 into a 4x4 frame: the ratio leaves a 4x2 content box with one black row above and
//! below, so one golden pins the downsample, the centring and the [0, 1] -> [-1, 1] shift.
TEST_F(Pi05PolicyTest, PreprocessedViewMatchesTheReferenceResizeWithPad)
{
    std::vector<uint8_t> const rgb{0, 32, 64, 16, 48, 80, 32, 64, 96, 48, 80, 112, 64, 96, 128, 128, 160, 192, 144, 176,
        208, 160, 192, 224, 176, 208, 240, 192, 224, 255, 255, 224, 192, 240, 208, 176, 224, 192, 160, 208, 176, 144,
        192, 160, 128};
    std::vector<float> planar(3U * 4U * 4U);
    pi05::resizeWithPad(rgb.data(), 3, 5, 4, 4, planar.data());

    std::vector<uint8_t> const golden{0, 0, 0, 0, 52, 71, 89, 108, 206, 202, 198, 193, 0, 0, 0, 0, 0, 0, 0, 0, 84, 103,
        121, 140, 199, 194, 190, 185, 0, 0, 0, 0, 0, 0, 0, 0, 116, 135, 153, 172, 191, 186, 182, 177, 0, 0, 0, 0};
    expectMatchesGolden(planar, golden);
}
