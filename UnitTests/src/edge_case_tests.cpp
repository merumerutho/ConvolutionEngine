
#include "test_common.h"

// Test ID: EDGE-01 - Block Size Variation
TEST_P(ConvolutionEngineTest, BlockSizeVariation) {
    // 1. Get baseline response with standard block size
    std::vector<float> baseline_response;
    {
        SetUp(); // Re-initialize everything
        dense_taps_data = {0.5f, 0.25f};
        DenseIRHandle handle;
        handle.taps = dense_taps_data.data();
        handle.num_taps = dense_taps_data.size();
        InitializeEngine(handle);
        input_buffer[0] = 1.0f;
        size_t total_samples = config.buffer_size;
        while(total_samples > 0) {
            size_t current_block = std::min(config.block_size, total_samples);
            engine->Process(input_buffer.data(), output_buffer.data(), current_block);
            baseline_response.insert(baseline_response.end(), output_buffer.begin(), output_buffer.begin() + current_block * config.num_channels);
            std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
            total_samples -= current_block;
        }
        TearDown();
    }

    // 2. Get response with block size = 1
    std::vector<float> single_sample_response;
    {
        SetUp(); // Re-initialize
        dense_taps_data = {0.5f, 0.25f};
        DenseIRHandle handle;
        handle.taps = dense_taps_data.data();
        handle.num_taps = dense_taps_data.size();
        InitializeEngine(handle);
        input_buffer[0] = 1.0f;
        size_t total_samples = config.buffer_size;
        while(total_samples > 0) {
            engine->Process(input_buffer.data(), output_buffer.data(), 1);
            single_sample_response.insert(single_sample_response.end(), output_buffer.begin(), output_buffer.begin() + 1 * config.num_channels);
            std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
            total_samples -= 1;
        }
        //TearDown();
    }

    // 3. Compare results
    ASSERT_EQ(baseline_response.size(), single_sample_response.size());
    for (size_t i = 0; i < baseline_response.size(); ++i) {
        ASSERT_NEAR(baseline_response[i], single_sample_response[i], 1e-6);
    }
}

// Test ID: EDGE-02 - Zero-Length IR
TEST_P(ConvolutionEngineTest, ZeroLengthIR) {
    DenseIRHandle handle;
    handle.num_taps = 0;
    handle.taps = nullptr;
    InitializeEngine(handle);

    input_buffer[0] = 1.0f;
    engine->Process(input_buffer.data(), output_buffer.data(), config.block_size);

    for(const auto& sample : output_buffer) {
        ASSERT_EQ(sample, 0.0f);
    }
}

// Test ID: EDGE-03 - Zero-Size Process Call
TEST_P(ConvolutionEngineTest, ZeroSizeProcessCall) {
    DenseIRHandle handle;
    dense_taps_data = {1.0f};
    handle.taps = dense_taps_data.data();
    handle.num_taps = dense_taps_data.size();
    InitializeEngine(handle);

    // Give the engine some state
    input_buffer[0] = 1.0f;
    engine->Process(input_buffer.data(), output_buffer.data(), config.block_size);

    // Now make a zero-size call
    auto initial_write_head = get_write_head();
    engine->Process(input_buffer.data(), output_buffer.data(), 0);

    EXPECT_EQ(get_write_head(), initial_write_head);
}
