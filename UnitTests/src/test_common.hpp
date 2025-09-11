
#pragma once

#include "gtest/gtest.h"
#include "../../DenseConvolutionEngine.hpp"
#include "../../SparseConvolutionEngine.hpp"
#include "../../VelvetConvolutionEngine.hpp"
#include "../../IRHandle.hpp"
#include <vector>
#include <numeric>
#include <memory>

#include <stdio.h>

// Test Configuration Struct
enum class IRType { DENSE, SPARSE, VELVET };
struct TestConfig {
    IRType ir_type;
    DenseConvolutionEngine::ChannelLayout channel_layout;
    size_t num_channels;
    size_t buffer_size;
    size_t block_size;
    DenseConvolutionEngine::WrappingMode wrapping_mode;

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

// Base interface for engine access
class EngineWrapper {
public:
    virtual ~EngineWrapper() = default;
    virtual void Process(const float* in, float* out, size_t size) = 0;
    virtual size_t get_write_head() const = 0;
    virtual size_t get_buffer_size() const = 0;
    virtual size_t get_num_channels() const = 0;
    virtual float* get_circ_buffer() const = 0;
    virtual void* get_active_process_function() const = 0;
};

template<typename EngineType>
class EngineWrapperImpl : public EngineWrapper {
public:
    EngineType engine;
    
    template<typename IRHandleType>
    void Init(IRHandleType& handle, float* buffer, size_t buffer_size, size_t num_channels) {
        engine.Init(handle, buffer, buffer_size, num_channels);
    }
    
    void Process(const float* in, float* out, size_t size) override {
        engine.Process(in, out, size);
    }
    
    size_t get_write_head() const override { return engine.write_head_; }
    size_t get_buffer_size() const override { return engine.buffer_size_; }
    size_t get_num_channels() const override { return engine.num_channels_; }
    float* get_circ_buffer() const override { return engine.circ_buffer_; }
    void* get_active_process_function() const override { return (void*)engine.active_process_function_; }
};

// The main test fixture
class ConvolutionEngineTest : public ::testing::TestWithParam<TestConfig> {
protected:
    void SetUp() override {
        config = GetParam();
        
        // Create appropriate engine based on config
        switch (config.ir_type) {
            case IRType::DENSE:
                engine_wrapper = std::make_unique<EngineWrapperImpl<DenseConvolutionEngine>>();
                break;
            case IRType::SPARSE:
                engine_wrapper = std::make_unique<EngineWrapperImpl<SparseConvolutionEngine>>();
                break;
            case IRType::VELVET:
                engine_wrapper = std::make_unique<EngineWrapperImpl<VelvetConvolutionEngine>>();
                break;
        }

        // Allocate memory for buffers based on config
        ir_buffer.resize(config.buffer_size, 0.0f);
        input_buffer.resize(config.block_size * config.num_channels, 0.0f);
        output_buffer.resize(config.block_size * config.num_channels, 0.0f);
    }

    void TearDown() override {
        engine_wrapper.reset();
    }

    void InitializeEngine(DenseIRHandle& handle) {
        if (config.ir_type == IRType::DENSE) {
            static_cast<EngineWrapperImpl<DenseConvolutionEngine>*>(engine_wrapper.get())->Init(handle, ir_buffer.data(), ir_buffer.size(), config.num_channels);
        }
    }
    
    void InitializeEngine(SparseIRHandle& handle) {
        if (config.ir_type == IRType::SPARSE) {
            static_cast<EngineWrapperImpl<SparseConvolutionEngine>*>(engine_wrapper.get())->Init(handle, ir_buffer.data(), ir_buffer.size(), config.num_channels);
        }
    }
    
    void InitializeEngine(VelvetIRHandle& handle) {
        if (config.ir_type == IRType::VELVET) {
            static_cast<EngineWrapperImpl<VelvetConvolutionEngine>*>(engine_wrapper.get())->Init(handle, ir_buffer.data(), ir_buffer.size(), config.num_channels);
        }
    }

    std::unique_ptr<EngineWrapper> engine_wrapper;
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

    // Helper getters for protected members
    size_t get_write_head() const { return engine_wrapper->get_write_head(); }
    size_t get_buffer_size() const { return engine_wrapper->get_buffer_size(); }
    size_t get_num_channels() const { return engine_wrapper->get_num_channels(); }
    float* get_circ_buffer() const { return engine_wrapper->get_circ_buffer(); }
    void* get_active_process_function() const { return engine_wrapper->get_active_process_function(); }
};
