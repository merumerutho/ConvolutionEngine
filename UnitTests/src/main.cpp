#include "gtest/gtest.h"
#include "test_common.hpp"

// Helper to generate a string name for each test case
inline std::string getTestName(const testing::TestParamInfo<TestConfig>& info) {
    std::string name;
    switch (info.param.ir_type) {
        case IRType::DENSE: name += "Dense"; break;
        case IRType::SPARSE: name += "Sparse"; break;
        case IRType::VELVET: name += "Velvet"; break;
    }
    name += "_";
    switch (info.param.channel_layout) {
        case DenseConvolutionEngine::ChannelLayout::MONO: name += "Mono"; break;
        case DenseConvolutionEngine::ChannelLayout::STEREO: name += "Stereo"; break;
        case DenseConvolutionEngine::ChannelLayout::QUAD: name += "Quad"; break;
        case DenseConvolutionEngine::ChannelLayout::MULTICHANNEL: name += "Multi"; break;
    }
    name += "_N" + std::to_string(info.param.num_channels);
    name += "_Buf" + std::to_string(info.param.buffer_size);
    name += "_Blk" + std::to_string(info.param.block_size);
    return name;
}

// Define a set of configurations to test against
const std::vector<TestConfig> test_configs = {
    // DENSE
    {IRType::DENSE, DenseConvolutionEngine::ChannelLayout::MONO, 1, 1024, 64, DenseConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::DENSE, DenseConvolutionEngine::ChannelLayout::STEREO, 2, 2048, 128, DenseConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::DENSE, DenseConvolutionEngine::ChannelLayout::MULTICHANNEL, 5, 4096, 256, DenseConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::DENSE, DenseConvolutionEngine::ChannelLayout::QUAD, 4, 2047, 64, DenseConvolutionEngine::WrappingMode::ARBITRARY}, 
    {IRType::DENSE, DenseConvolutionEngine::ChannelLayout::MONO, 1, 511, 16, DenseConvolutionEngine::WrappingMode::ARBITRARY}, 

    // SPARSE
    {IRType::SPARSE, SparseConvolutionEngine::ChannelLayout::MONO, 1, 1024, 64, SparseConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::SPARSE, SparseConvolutionEngine::ChannelLayout::STEREO, 2, 2048, 128, SparseConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::SPARSE, SparseConvolutionEngine::ChannelLayout::MULTICHANNEL, 6, 4096, 256, SparseConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::SPARSE, SparseConvolutionEngine::ChannelLayout::QUAD, 4, 2047, 128, SparseConvolutionEngine::WrappingMode::ARBITRARY},  
    {IRType::SPARSE, SparseConvolutionEngine::ChannelLayout::STEREO, 2, 2047, 64, SparseConvolutionEngine::WrappingMode::ARBITRARY},  

    // VELVET
    {IRType::VELVET, VelvetConvolutionEngine::ChannelLayout::MONO, 1, 1024, 64, VelvetConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::VELVET, VelvetConvolutionEngine::ChannelLayout::STEREO, 2, 2048, 128, VelvetConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::VELVET, VelvetConvolutionEngine::ChannelLayout::MULTICHANNEL, 3, 4096, 256, VelvetConvolutionEngine::WrappingMode::POWER_OF_TWO},
    {IRType::VELVET, VelvetConvolutionEngine::ChannelLayout::QUAD, 4, 2047, 4, VelvetConvolutionEngine::WrappingMode::ARBITRARY}, 
    {IRType::VELVET, VelvetConvolutionEngine::ChannelLayout::MULTICHANNEL, 7, 4095, 32, VelvetConvolutionEngine::WrappingMode::ARBITRARY}, 
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
