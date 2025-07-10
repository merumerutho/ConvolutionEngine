
#pragma once

#include "gtest/gtest.h"
#include "../../ConvolutionEngine.h"
#include "../../IRHandle.h"
#include <vector>
#include <numeric>

#include <stdio.h>

// Test Configuration Struct
struct TestConfig {
    ConvolutionEngine::IRType ir_type;
    ConvolutionEngine::ChannelLayout channel_layout;
    size_t num_channels;
    size_t buffer_size;
    size_t block_size;
    ConvolutionEngine::WrappingMode wrapping_mode;

    // For nice test names
    friend std::ostream& operator<<(std::ostream& os, const TestConfig& config) {
        os << "IR" << static_cast<int>(config.ir_type)
           << "_CH" << static_cast<int>(config.channel_layout)
           << "_N" << config.num_channels
           << "_BUF" << config.buffer_size
           << "_BLK" << config.block_size;
        return os;
    }
};

// The main test fixture
class ConvolutionEngineTest : public ::testing::TestWithParam<TestConfig> {
protected:
    void SetUp() override {
        config = GetParam();
        engine = new ConvolutionEngine();

        // Allocate memory for buffers based on config
        ir_buffer.resize(config.buffer_size, 0.0f);
        input_buffer.resize(config.block_size * config.num_channels, 0.0f);
        output_buffer.resize(config.block_size * config.num_channels, 0.0f);
    }

    void TearDown() override {
        delete engine;
    }

    template<typename IRHandleType>
    void InitializeEngine(IRHandleType& handle) {
        handle.num_channels = config.num_channels;
        engine->Init(handle, ir_buffer.data(), ir_buffer.size());
    }

    ConvolutionEngine* engine;
    TestConfig config;

    // Buffers
    std::vector<float> ir_buffer;
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;

    // IR Data
    std::vector<float> dense_taps_data;
    std::vector<size_t> sparse_positions_data;
    std::vector<float> sparse_values_data;
    std::vector<size_t> velvet_pos_taps_data;
    std::vector<size_t> velvet_neg_taps_data;

    // Helper getters for protected members of ConvolutionEngine
    size_t get_write_head() const { return engine->write_head_; }
    size_t get_buffer_size() const { return engine->buffer_size_; }
    size_t get_num_channels() const { return engine->num_channels_; }
    float* get_circ_buffer() const { return engine->circ_buffer_; }
    ConvolutionEngine::ProcessFunctionPtr get_active_process_function() const { return engine->active_process_function_; }
};
