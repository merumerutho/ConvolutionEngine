#pragma once
#ifndef DENSE_CONVOLUTION_ENGINE_H
#define DENSE_CONVOLUTION_ENGINE_H

#include "IRHandle.hpp"
#include "ConvolutionUtils.hpp"
#include <cstddef>   
#include <algorithm> 
#include <cmath>     
#include <type_traits> 

/**
 * @brief A generic, real-time safe convolution engine supporting dense
 * impulse responses with compile-time optimization for channel layout.
 */
class DenseConvolutionEngine
{
    friend class ConvolutionEngineTest;
    template<typename EngineType> friend class EngineWrapperImpl;
public:
    using WrappingMode = ConvolutionUtils::WrappingMode;
    using ChannelLayout = ConvolutionUtils::ChannelLayout;

    using ProcessFunctionPtr = void (DenseConvolutionEngine::*)(const float*, float*, size_t);

    DenseConvolutionEngine() : active_process_function_(nullptr), is_morphing_(false), 
                               morph_cycles_remaining_(0), morph_delta_(nullptr) {}
    ~DenseConvolutionEngine() {}

    void Init(const DenseIRHandle& handle, float* circ_buffer, size_t buffer_size, size_t num_channels,
              float* current_taps_buffer = nullptr, float* morph_delta_buffer = nullptr)
    {
        circ_buffer_  = circ_buffer;
        buffer_size_  = buffer_size;
        num_channels_ = num_channels;
        write_head_   = 0;

        assert(circ_buffer_); 	// != NULL
        assert(buffer_size);  	// > 0 
        assert(num_channels_); 	// > 0

		// Init empty buffer
        std::fill(circ_buffer_, circ_buffer_ + buffer_size_, 0.0f);

        const bool is_pow2 = ConvolutionUtils::is_power_of_two(buffer_size_);
        const auto wrapping_mode = is_pow2 ? WrappingMode::POWER_OF_TWO : WrappingMode::ARBITRARY;

        dense_taps_ 				= handle.taps;
        num_dense_taps_ 			= handle.num_taps;
        assert(dense_taps_          || (num_dense_taps_ == 0));         // != NULL

        // Initialize morphing buffers if provided
        current_taps_ = current_taps_buffer;
        morph_delta_ = morph_delta_buffer;
        
        if (current_taps_ && num_dense_taps_ > 0)
        {
            // Copy initial IR values to current_taps_ buffer
            std::copy(dense_taps_, dense_taps_ + num_dense_taps_, current_taps_);
        }

        // Dispatch to the correct template specialization based on runtime channel count
        // and WrappingMode
        if (num_channels_ == 1)
		{
            dispatch_set_process_function(wrapping_mode, ChannelLayout::MONO);
		}
        else if (num_channels_ == 2)
		{
            dispatch_set_process_function(wrapping_mode, ChannelLayout::STEREO);
		}
        else if (num_channels_ == 4)
		{
            dispatch_set_process_function(wrapping_mode, ChannelLayout::QUAD);
		}
        else
		{
            dispatch_set_process_function(wrapping_mode, ChannelLayout::MULTICHANNEL);
		}
    }

    void Process(const float* in, float* out, size_t size)
    {
        (this->*active_process_function_)(in, out, size);
    }

    void MorphIRDense(const DenseIRHandle& target_handle, int morph_cycles)
    {
        assert(morph_cycles > 0);
        assert(target_handle.taps);
        assert(target_handle.num_taps == num_dense_taps_);
        assert(current_taps_);
        assert(morph_delta_);

        target_taps_ = target_handle.taps;
        morph_cycles_remaining_ = morph_cycles;
        
        // Calculate deltas for each tap (from current state to target)
        for (size_t i = 0; i < num_dense_taps_; i++)
        {
            morph_delta_[i] = (target_taps_[i] - current_taps_[i]) / static_cast<float>(morph_cycles);
        }
        
        // Switch to using current_taps_ for processing
        dense_taps_ = current_taps_;
        is_morphing_ = true;
    }

    void MorphIRDense_Update()
    {
        if (!is_morphing_ || morph_cycles_remaining_ <= 0)
            return;

        // Apply one step of linear interpolation
        for (size_t i = 0; i < num_dense_taps_; i++)
        {
            current_taps_[i] += morph_delta_[i];
        }

        morph_cycles_remaining_--;
        
        if (morph_cycles_remaining_ == 0)
        {
            // Morphing complete - snap to target values and update main pointer
            for (size_t i = 0; i < num_dense_taps_; i++)
            {
                current_taps_[i] = target_taps_[i];
            }
            dense_taps_ = current_taps_;
            is_morphing_ = false;
        }
    }

protected:
    template <WrappingMode WMode, ChannelLayout CLayout>
    void set_process_function(WrappingMode)
    {
        active_process_function_ = &DenseConvolutionEngine::ProcessImpl<WMode, CLayout>;
    }

    void dispatch_set_process_function(WrappingMode wrapping_mode, ChannelLayout channel_layout)
    {
		// POWER OF TWO
        if (wrapping_mode == WrappingMode::POWER_OF_TWO) {
            if (channel_layout == ChannelLayout::MONO) set_process_function<WrappingMode::POWER_OF_TWO, ChannelLayout::MONO>(wrapping_mode);
            else if (channel_layout == ChannelLayout::STEREO) set_process_function<WrappingMode::POWER_OF_TWO, ChannelLayout::STEREO>(wrapping_mode);
            else if (channel_layout == ChannelLayout::QUAD) set_process_function<WrappingMode::POWER_OF_TWO, ChannelLayout::QUAD>(wrapping_mode);
            else set_process_function<WrappingMode::POWER_OF_TWO, ChannelLayout::MULTICHANNEL>(wrapping_mode);
        } else { // ARBITRARY
            if (channel_layout == ChannelLayout::MONO) set_process_function<WrappingMode::ARBITRARY, ChannelLayout::MONO>(wrapping_mode);
            else if (channel_layout == ChannelLayout::STEREO) set_process_function<WrappingMode::ARBITRARY, ChannelLayout::STEREO>(wrapping_mode);
            else if (channel_layout == ChannelLayout::QUAD) set_process_function<WrappingMode::ARBITRARY, ChannelLayout::QUAD>(wrapping_mode);
            else set_process_function<WrappingMode::ARBITRARY, ChannelLayout::MULTICHANNEL>(wrapping_mode);
        }
    }


    template <WrappingMode WMode, ChannelLayout CLayout>
    void ProcessImpl(const float* in, float* out, size_t size)
    {
        const size_t num_ch = num_channels_;

        for (size_t smp = 0; smp < size; smp++)
        {
            // DENSE KERNEL
            for (size_t t = 0; t < num_dense_taps_; t++)
                ConvolutionUtils::for_each_channel<CLayout>([&](size_t ch) 
                {
                    const size_t tap_pos = ConvolutionUtils::wrap_address<WMode>(write_head_ + ch + t * num_ch, buffer_size_);
                    circ_buffer_[tap_pos] += in[smp*num_ch + ch] * dense_taps_[t];
                }, num_channels_);
            
            // Extract output and clear buffer
            ConvolutionUtils::for_each_channel<CLayout>([&](size_t ch) {
                out[smp * num_ch + ch] = circ_buffer_[ConvolutionUtils::wrap_address<WMode>(write_head_ + ch, buffer_size_)];
				circ_buffer_[ConvolutionUtils::wrap_address<WMode>(write_head_ + ch, buffer_size_)] = 0.0f;
            }, num_channels_);
			
			// Advance buffer head
			write_head_ = ConvolutionUtils::wrap_address<WMode>(write_head_ + num_ch, buffer_size_);
        }
    }


    ProcessFunctionPtr active_process_function_;

    size_t write_head_;
    size_t buffer_size_;
    size_t num_channels_;
    float* circ_buffer_;

	// Used for DENSE kernel
    const float * dense_taps_;
    size_t        num_dense_taps_;

    // Morphing state
    bool is_morphing_;
    int morph_cycles_remaining_;
    const float* target_taps_;
    float* current_taps_;
    float* morph_delta_;
};

#endif // DENSE_CONVOLUTION_ENGINE_H