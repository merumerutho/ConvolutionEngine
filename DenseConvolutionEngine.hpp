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

    DenseConvolutionEngine() : active_process_function_(nullptr) {}
    ~DenseConvolutionEngine() {}

    void Init(const DenseIRHandle& handle, float* circ_buffer, size_t buffer_size, size_t num_channels)
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
};

#endif // DENSE_CONVOLUTION_ENGINE_H