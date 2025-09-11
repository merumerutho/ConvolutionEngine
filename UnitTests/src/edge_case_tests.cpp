#include "test_common.hpp"

// Test ID: EDGE-01 - Block Size Variation
TEST_P(ConvolutionEngineTest, BlockSizeVariation) {
    auto initializeTestEngine = [this]() {
        switch (config.ir_type) {
            case IRType::DENSE: {
                dense_taps_data = {0.5f, 0.25f};
                DenseIRHandle handle;
                handle.taps = dense_taps_data.data();
                handle.num_taps = dense_taps_data.size();
                InitializeEngine(handle);
                break;
            }
            case IRType::SPARSE: {
                sparse_positions_data = {0, 1};
                sparse_values_data = {0.5f, 0.25f};
                SparseIRHandle handle;
                handle.positions = sparse_positions_data.data();
                handle.values = sparse_values_data.data();
                handle.num_taps = sparse_positions_data.size();
                InitializeEngine(handle);
                break;
            }
            case IRType::VELVET: {
                velvet_pos_taps_data = {0, 1};
                velvet_neg_taps_data = {};
                VelvetIRHandle handle;
                handle.pos_taps = velvet_pos_taps_data.data();
                handle.num_pos_taps = velvet_pos_taps_data.size();
                handle.neg_taps = nullptr;
                handle.num_neg_taps = 0;
                InitializeEngine(handle);
                break;
            }
        }
    };
    
    // 1. Get baseline response with standard block size
    std::vector<float> baseline_response;
    {
        SetUp(); // Re-initialize everything
        initializeTestEngine();
        input_buffer[0] = 1.0f;
        size_t total_samples = config.buffer_size;
        while(total_samples > 0) {
            size_t current_block = std::min(config.block_size, total_samples);
            engine_wrapper->Process(input_buffer.data(), output_buffer.data(), current_block);
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
        initializeTestEngine();
        input_buffer[0] = 1.0f;
        size_t total_samples = config.buffer_size;
        while(total_samples > 0) {
            engine_wrapper->Process(input_buffer.data(), output_buffer.data(), 1);
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
    switch (config.ir_type) {
        case IRType::DENSE: {
            DenseIRHandle handle;
            handle.num_taps = 0;
            handle.taps = nullptr;
            InitializeEngine(handle);
            break;
        }
        case IRType::SPARSE: {
            SparseIRHandle handle;
            handle.num_taps = 0;
            handle.positions = nullptr;
            handle.values = nullptr;
            InitializeEngine(handle);
            break;
        }
        case IRType::VELVET: {
            VelvetIRHandle handle;
            handle.num_pos_taps = 0;
            handle.pos_taps = nullptr;
            handle.num_neg_taps = 0;
            handle.neg_taps = nullptr;
            InitializeEngine(handle);
            break;
        }
    }

    input_buffer[0] = 1.0f;
    engine_wrapper->Process(input_buffer.data(), output_buffer.data(), config.block_size);

    for(const auto& sample : output_buffer) {
        ASSERT_EQ(sample, 0.0f);
    }
}

// Test ID: EDGE-03 - Zero-Size Process Call
TEST_P(ConvolutionEngineTest, ZeroSizeProcessCall) {
    switch (config.ir_type) {
        case IRType::DENSE: {
            DenseIRHandle handle;
            dense_taps_data = {1.0f};
            handle.taps = dense_taps_data.data();
            handle.num_taps = dense_taps_data.size();
            InitializeEngine(handle);
            break;
        }
        case IRType::SPARSE: {
            SparseIRHandle handle;
            sparse_positions_data = {0};
            sparse_values_data = {1.0f};
            handle.positions = sparse_positions_data.data();
            handle.values = sparse_values_data.data();
            handle.num_taps = sparse_positions_data.size();
            InitializeEngine(handle);
            break;
        }
        case IRType::VELVET: {
            VelvetIRHandle handle;
            velvet_pos_taps_data = {0};
            handle.pos_taps = velvet_pos_taps_data.data();
            handle.num_pos_taps = velvet_pos_taps_data.size();
            handle.neg_taps = nullptr;
            handle.num_neg_taps = 0;
            InitializeEngine(handle);
            break;
        }
    }

    // Give the engine some state
    input_buffer[0] = 1.0f;
    engine_wrapper->Process(input_buffer.data(), output_buffer.data(), config.block_size);

    // Now make a zero-size call
    auto initial_write_head = get_write_head();
    engine_wrapper->Process(input_buffer.data(), output_buffer.data(), 0);

    EXPECT_EQ(get_write_head(), initial_write_head);
}
