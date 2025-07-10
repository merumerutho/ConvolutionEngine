#include "test_common.hpp"

// Test ID: FUNC-01 - Impulse Response
TEST_P(ConvolutionEngineTest, ImpulseResponse) {
    // 1. Initialize based on IR type
    switch (config.ir_type) {
        case ConvolutionEngine::IRType::DENSE: {
            dense_taps_data = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f};
            DenseIRHandle handle;
            handle.taps = dense_taps_data.data();
            handle.num_taps = dense_taps_data.size();
            InitializeEngine(handle);
            break;
        }
        case ConvolutionEngine::IRType::SPARSE: {
            sparse_positions_data = {10, 20, 30};
            sparse_values_data = {0.8f, 0.7f, 0.6f};
            SparseIRHandle handle;
            handle.positions = sparse_positions_data.data();
            handle.values = sparse_values_data.data();
            handle.num_taps = sparse_positions_data.size();
            InitializeEngine(handle);
            break;
        }
        case ConvolutionEngine::IRType::VELVET: {
            velvet_pos_taps_data = {5, 15};
            velvet_neg_taps_data = {25};
            VelvetIRHandle handle;
            handle.pos_taps = velvet_pos_taps_data.data();
            handle.num_pos_taps = velvet_pos_taps_data.size();
            handle.neg_taps = velvet_neg_taps_data.data();
            handle.num_neg_taps = velvet_neg_taps_data.size();
            InitializeEngine(handle);
            break;
        }
    }

    // 2. Prepare impulse
    // Trick to verify multich correctness: make impulse get amplitude based on channel
    for (size_t ch = 0; ch < config.num_channels; ++ch) {
        input_buffer[ch] = ch*1.0f; // Impulse on the first sample of each channel
    }

    // 3. Process and collect output
    std::vector<float> full_response;
    size_t total_samples_to_process = config.buffer_size;
    while (total_samples_to_process > 0) {
        size_t current_block_size = std::min((size_t)config.block_size, total_samples_to_process);
        engine->Process(input_buffer.data(), output_buffer.data(), current_block_size);
        full_response.insert(full_response.end(), output_buffer.begin(), output_buffer.begin() + (current_block_size * config.num_channels));
        // After the first block, the input should be silence
        std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
        total_samples_to_process -= current_block_size;
    }

    // 4. Verify the output
    for (size_t ch = 0; ch < config.num_channels; ch++) 
    {
        size_t expected_amplitude = ch;

        switch (config.ir_type) {
            case ConvolutionEngine::IRType::DENSE: {
                for (size_t i = 0; i < dense_taps_data.size(); i++) 
                {
                    float expected_sample = expected_amplitude * dense_taps_data[i];
                    ASSERT_NEAR(full_response[i * config.num_channels + ch], expected_sample, 1e-6);
                }
                break;
            }
            case ConvolutionEngine::IRType::SPARSE: {
                for (size_t i = 0; i < sparse_positions_data.size(); i++) 
                {
                    float expected_sample = expected_amplitude*sparse_values_data[i];
                    ASSERT_NEAR(full_response[sparse_positions_data[i] * config.num_channels + ch], expected_sample, 1e-6);
                }
                break;
            }
            case ConvolutionEngine::IRType::VELVET: {
                for (auto& pos : velvet_pos_taps_data) 
                {
                    float expected_sample = expected_amplitude*1.0f;
                    ASSERT_NEAR(full_response[pos * config.num_channels + ch], expected_sample, 1e-6);
                }
                for (auto& pos : velvet_neg_taps_data) 
                {
                    float expected_sample = expected_amplitude * (-1.0f);
                    ASSERT_NEAR(full_response[pos * config.num_channels + ch], expected_sample, 1e-6);
                }
                break;
            }
        }
    }
}

// Test ID: FUNC-03 - Response to Silence
TEST_P(ConvolutionEngineTest, SilenceResponse) {
    // Initialize with any valid handle
    DenseIRHandle handle;
    dense_taps_data = {1.0f};
    handle.taps = dense_taps_data.data();
    handle.num_taps = dense_taps_data.size();
    InitializeEngine(handle);

    std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
    engine->Process(input_buffer.data(), output_buffer.data(), config.block_size);

    for(const auto& sample : output_buffer) {
        ASSERT_EQ(sample, 0.0f);
    }
}
