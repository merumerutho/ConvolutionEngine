#include "test_common.h"

// Test ID: BUFF-02 - Write Head Advancement
TEST_P(ConvolutionEngineTest, WriteHeadAdvancement) {
    DenseIRHandle handle; // Simple handle
    handle.num_taps = 0;
    handle.taps = nullptr;
	
    InitializeEngine(handle);

    size_t initial_write_head = get_write_head();
    size_t samples_to_process = config.block_size;

    size_t n_blocks = (size_t) std::ceil((float)(config.buffer_size) / config.block_size);
    
    // Check that write head is increasing until it wraps back to 0 
    for (size_t i = 0; i < n_blocks; i++)
    {
        size_t expected_current_pos = (initial_write_head + config.block_size*i*config.num_channels) % config.buffer_size;
        size_t current_head = get_write_head();
        EXPECT_EQ(current_head, expected_current_pos);
        engine->Process(input_buffer.data(), output_buffer.data(), samples_to_process);
    }
    // Last check 
    size_t expected_current_pos = (initial_write_head + config.block_size*n_blocks*config.num_channels) % config.buffer_size;
    size_t current_head = get_write_head();
    EXPECT_EQ(current_head, expected_current_pos);

    // The write head should wrap around to its starting position
    // Process one more sample
    engine->Process(input_buffer.data(), output_buffer.data(), 1);
    size_t expected_head = (current_head + config.num_channels) % config.buffer_size;
	
    EXPECT_EQ(get_write_head(), expected_head);
}

