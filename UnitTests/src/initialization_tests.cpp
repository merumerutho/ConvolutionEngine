#include "test_common.hpp"

// Test ID: INIT-01 - Default State
TEST_P(ConvolutionEngineTest, DefaultState) {
    // The friend class allows us to access the protected member.
    EXPECT_EQ(get_active_process_function(), nullptr);
}

// Test ID: INIT-02 - Post-Init State and Dispatch
TEST_P(ConvolutionEngineTest, PostInitState) {
    switch (config.ir_type) {
        case ConvolutionEngine::IRType::DENSE: {
            DenseIRHandle handle;
            handle.num_taps = 0;
            handle.taps = nullptr;
            InitializeEngine(handle);
            break;
        }
        case ConvolutionEngine::IRType::SPARSE: {
            SparseIRHandle handle;
            handle.num_taps = 0;
            handle.positions = nullptr;
            handle.values = nullptr;
            InitializeEngine(handle);
            break;
        }
        case ConvolutionEngine::IRType::VELVET: {
            VelvetIRHandle handle;
            handle.num_pos_taps = 0;
            handle.pos_taps = nullptr;
            handle.num_neg_taps = 0;
            handle.neg_taps = nullptr;
            InitializeEngine(handle);
            break;
        }
    }

    // Verify internal state
    EXPECT_EQ(get_buffer_size(), config.buffer_size);
    EXPECT_EQ(get_num_channels(), config.num_channels);
    EXPECT_EQ(get_write_head(), 0);

    // Verify buffer is cleared
    for (size_t i = 0; i < config.buffer_size; ++i) {
        EXPECT_EQ(get_circ_buffer()[i], 0.0f);
    }
}

