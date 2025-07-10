#include "gtest/gtest.h"
#include "test_common.hpp"

// Helper to generate a string name for each test case
inline std::string getTestName(const testing::TestParamInfo<TestConfig>& info) {
    std::string name;
    switch (info.param.ir_type) {
        case ConvolutionEngine::IRType::DENSE: name += "Dense"; break;
        case ConvolutionEngine::IRType::SPARSE: name += "Sparse"; break;
        case ConvolutionEngine::IRType::VELVET: name += "Velvet"; break;
    }
    name += "_";
    switch (info.param.channel_layout) {
        case ConvolutionEngine::ChannelLayout::MONO: name += "Mono"; break;
        case ConvolutionEngine::ChannelLayout::STEREO: name += "Stereo"; break;
        case ConvolutionEngine::ChannelLayout::QUAD: name += "Quad"; break;
        case ConvolutionEngine::ChannelLayout::MULTICHANNEL: name += "Multi"; break;
    }
    name += "_N" + std::to_string(info.param.num_channels);
    name += "_Buf" + std::to_string(info.param.buffer_size);
    name += "_Blk" + std::to_string(info.param.block_size);
    return name;
}

// Define a set of configurations to test against
const std::vector<TestConfig> test_configs = {
    // DENSE
    {ConvolutionEngine::IRType::DENSE, ConvolutionEngine::ChannelLayout::MONO, 1, 1024, 64, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::DENSE, ConvolutionEngine::ChannelLayout::STEREO, 2, 2048, 128, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::DENSE, ConvolutionEngine::ChannelLayout::MULTICHANNEL, 5, 4096, 256, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::DENSE, ConvolutionEngine::ChannelLayout::QUAD, 4, 2047, 64, ConvolutionEngine::WrappingMode::ARBITRARY}, 
    {ConvolutionEngine::IRType::DENSE, ConvolutionEngine::ChannelLayout::MONO, 1, 511, 16, ConvolutionEngine::WrappingMode::ARBITRARY}, 

    // SPARSE
    {ConvolutionEngine::IRType::SPARSE, ConvolutionEngine::ChannelLayout::MONO, 1, 1024, 64, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::SPARSE, ConvolutionEngine::ChannelLayout::STEREO, 2, 2048, 128, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::SPARSE, ConvolutionEngine::ChannelLayout::MULTICHANNEL, 6, 4096, 256, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::SPARSE, ConvolutionEngine::ChannelLayout::QUAD, 4, 2047, 128, ConvolutionEngine::WrappingMode::ARBITRARY},  
    {ConvolutionEngine::IRType::SPARSE, ConvolutionEngine::ChannelLayout::STEREO, 2, 2047, 64, ConvolutionEngine::WrappingMode::ARBITRARY},  

    // VELVET
    {ConvolutionEngine::IRType::VELVET, ConvolutionEngine::ChannelLayout::MONO, 1, 1024, 64, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::VELVET, ConvolutionEngine::ChannelLayout::STEREO, 2, 2048, 128, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::VELVET, ConvolutionEngine::ChannelLayout::MULTICHANNEL, 3, 4096, 256, ConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {ConvolutionEngine::IRType::VELVET, ConvolutionEngine::ChannelLayout::QUAD, 4, 2047, 4, ConvolutionEngine::WrappingMode::ARBITRARY}, 
    {ConvolutionEngine::IRType::VELVET, ConvolutionEngine::ChannelLayout::MULTICHANNEL, 7, 4095, 32, ConvolutionEngine::WrappingMode::ARBITRARY}, 
};

// Define the test configurations
INSTANTIATE_TEST_SUITE_P(
    ConvolutionEngineComprehensiveTests,
    ConvolutionEngineTest,
    ::testing::ValuesIn(test_configs),
    getTestName
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
	int dummy_variable = 0;
}
