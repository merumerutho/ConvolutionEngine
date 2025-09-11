#include "test_common.hpp"
#include "generated_test_data.hpp"

TEST_P(ConvolutionEngineTest, DenseIR) {
    // Only test dense-specific functionality when using Dense engine
    if (config.ir_type != IRType::DENSE) {
        return; // Silently return for non-Dense configurations
    }
    
    DenseIRHandle handle;
    handle.taps = dense_ir;
    handle.num_taps = dense_ir_size;
    InitializeEngine(handle);
	
	size_t n_blocks = (size_t) (input_signal_size / config.block_size);

    for (int i = 0; i < n_blocks; i++) 
	{
		// Prepare input
        for (size_t ch = 0; ch < config.num_channels; ch++) 
		{
            for (size_t j = 0; j < config.block_size; j++) 
			{
                input_buffer[j * config.num_channels + ch] = input_signal[config.block_size * i + j];
            }
        }
		
		// Process
        engine_wrapper->Process(input_buffer.data(), output_buffer.data(), config.block_size);
		ASSERT_EQ(get_write_head(), ((i+1)*config.block_size*config.num_channels) % config.buffer_size);
		
		// Evaluate output
		for (size_t ch = 0; ch < config.num_channels; ch++)
		{
			for (size_t j = 0; j < config.block_size; j++)
			{
				ASSERT_NEAR(output_buffer[j * config.num_channels + ch], expected_output_dense[config.block_size * i + j], 1e-3 );
			}
		}
		
    }
}

TEST_P(ConvolutionEngineTest, SparseIR) {
    // Only test sparse-specific functionality when using Sparse engine
    if (config.ir_type != IRType::SPARSE) {
        return; // Silently return for non-Sparse configurations
    }
    
    SparseIRHandle handle;
    handle.positions = sparse_ir_positions;
    handle.values = sparse_ir_values;
    handle.num_taps = sparse_ir_positions_size;
    InitializeEngine(handle);
	
    size_t total_samples_to_process = input_signal_size;
    size_t input_idx = 0;
	
	size_t n_blocks = input_signal_size / config.block_size;

    for (int i = 0; i < n_blocks; i++) {
        size_t current_block_size = std::min((size_t)config.block_size, total_samples_to_process);
        if (current_block_size == 0) break;

        for (size_t ch = 0; ch < config.num_channels; ch++) {
            for (size_t j = 0; j < current_block_size; j++) {
                if (input_idx + j < input_signal_size) {
                    input_buffer[j*config.num_channels + ch] = input_signal[input_idx + j];
                } else {
                    input_buffer[j*config.num_channels + ch] = 0.0f;
                }
            }
        }
		
        input_idx += current_block_size;
        engine_wrapper->Process(input_buffer.data(), output_buffer.data(), current_block_size);
		
		for (size_t ch = 0; ch < config.num_channels; ch++)
		{
			for (size_t j = 0; j < current_block_size; j++)
			{
				ASSERT_NEAR(output_buffer[j*config.num_channels + ch], expected_output_sparse[config.block_size * i + j], 1e-3 );
			}
		}
        total_samples_to_process -= current_block_size;
    }
}

TEST_P(ConvolutionEngineTest, VelvetIR) {
    // Only test velvet-specific functionality when using Velvet engine
    if (config.ir_type != IRType::VELVET) {
        return; // Silently return for non-Velvet configurations
    }
    
    VelvetIRHandle handle;
    handle.pos_taps = velvet_ir_pos_positions;
    handle.num_pos_taps = velvet_ir_pos_positions_size;
    handle.neg_taps = velvet_ir_neg_positions;
    handle.num_neg_taps = velvet_ir_neg_positions_size;
    InitializeEngine(handle);

    std::vector<float> full_response;
    size_t total_samples_to_process = input_signal_size;
    size_t input_idx = 0;

	size_t n_blocks = input_signal_size / config.block_size;

    for (int i = 0; i < n_blocks; i++) {
        size_t current_block_size = std::min((size_t)config.block_size, total_samples_to_process);
        if (current_block_size == 0) break;

        for (size_t ch = 0; ch < config.num_channels; ch++) {
            for (size_t j = 0; j < current_block_size; j++) {
                if (input_idx + j < input_signal_size) {
                    input_buffer[j * config.num_channels + ch] = input_signal[input_idx + j];
                } else {
                    input_buffer[j * config.num_channels + ch] = 0.0f;
                }
            }
        }
        input_idx += current_block_size;
        engine_wrapper->Process(input_buffer.data(), output_buffer.data(), current_block_size);
		
		for (size_t ch = 0; ch < config.num_channels; ch++)
		{
			for (size_t j = 0; j < current_block_size; j++)
			{
				ASSERT_NEAR(output_buffer[j*config.num_channels + ch], expected_output_velvet[config.block_size * i + j], 1e-3 );
			}
		}
    }
}


