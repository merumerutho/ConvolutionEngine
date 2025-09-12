#pragma once
#ifndef VELVET_CONVOLUTION_ENGINE_H
#define VELVET_CONVOLUTION_ENGINE_H

#include "IRHandle.hpp"
#include "ConvolutionUtils.hpp"
#include <cstddef>   
#include <algorithm> 
#include <cmath>     
#include <type_traits> 

/**
 * @brief A generic, real-time safe convolution engine supporting velvet
 * impulse responses with compile-time optimization for channel layout.
 */
class VelvetConvolutionEngine
{
    friend class ConvolutionEngineTest;
    template<typename EngineType> friend class EngineWrapperImpl;
public:
    using WrappingMode = ConvolutionUtils::WrappingMode;
    using ChannelLayout = ConvolutionUtils::ChannelLayout;

    using ProcessFunctionPtr = void (VelvetConvolutionEngine::*)(const float*, float*, size_t);

    VelvetConvolutionEngine() : active_process_function_(nullptr), is_morphing_(false),
                                initial_pos_tail_(0), initial_neg_tail_(0), 
                                target_pos_head_(0), target_neg_head_(0) {}
    ~VelvetConvolutionEngine() {}

    void Init(const VelvetIRHandle& handle, float* circ_buffer, size_t buffer_size, size_t num_channels,
              size_t* current_pos_buffer = nullptr, size_t* current_neg_buffer = nullptr,
              size_t* initial_pos_buffer = nullptr, size_t* initial_neg_buffer = nullptr,
              size_t* target_pos_buffer = nullptr, size_t* target_neg_buffer = nullptr,
              size_t max_pos_taps = 0, size_t max_neg_taps = 0)
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

        velvet_pos_taps_ 			= handle.pos_taps;
        num_velvet_pos_taps_ 		= handle.num_pos_taps;
        velvet_neg_taps_ 			= handle.neg_taps;
        num_velvet_neg_taps_ 		= handle.num_neg_taps;
        assert(velvet_pos_taps_     || (num_velvet_pos_taps_ == 0));    // != NULL
        assert(velvet_neg_taps_     || (num_velvet_neg_taps_ == 0));    // != NULL

        // Initialize morphing buffers if provided
        current_pos_taps_ = current_pos_buffer;
        current_neg_taps_ = current_neg_buffer;
        initial_pos_taps_ = initial_pos_buffer;
        initial_neg_taps_ = initial_neg_buffer;
        target_pos_taps_ = target_pos_buffer;
        target_neg_taps_ = target_neg_buffer;
        max_pos_taps_ = max_pos_taps;
        max_neg_taps_ = max_neg_taps;

        // Copy initial IR to current buffers if morphing is enabled
        if (current_pos_taps_ && num_velvet_pos_taps_ > 0)
        {
            std::copy(velvet_pos_taps_, velvet_pos_taps_ + num_velvet_pos_taps_, current_pos_taps_);
            num_current_pos_taps_ = num_velvet_pos_taps_;
        }
        else
        {
            num_current_pos_taps_ = 0;
        }

        if (current_neg_taps_ && num_velvet_neg_taps_ > 0)
        {
            std::copy(velvet_neg_taps_, velvet_neg_taps_ + num_velvet_neg_taps_, current_neg_taps_);
            num_current_neg_taps_ = num_velvet_neg_taps_;
        }
        else
        {
            num_current_neg_taps_ = 0;
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

    bool IsMorphing() const
    {
        return is_morphing_;
    }

    void MorphIRVelvet(const VelvetIRHandle& target_handle)
    {
        assert(current_pos_taps_);
        assert(current_neg_taps_);
        assert(initial_pos_taps_);
        assert(initial_neg_taps_);
        assert(target_pos_taps_);
        assert(target_neg_taps_);
        assert(target_handle.num_pos_taps <= max_pos_taps_);
        assert(target_handle.num_neg_taps <= max_neg_taps_);

        // Copy initial handle to working buffer (the one that gets consumed)
        std::copy(velvet_pos_taps_, velvet_pos_taps_ + num_velvet_pos_taps_, initial_pos_taps_);
        num_initial_pos_taps_ = num_velvet_pos_taps_;
        initial_pos_tail_ = (num_initial_pos_taps_ > 0) ? num_initial_pos_taps_ - 1 : 0;  // Start from end

        std::copy(velvet_neg_taps_, velvet_neg_taps_ + num_velvet_neg_taps_, initial_neg_taps_);
        num_initial_neg_taps_ = num_velvet_neg_taps_;
        initial_neg_tail_ = (num_initial_neg_taps_ > 0) ? num_initial_neg_taps_ - 1 : 0;  // Start from end

        // Copy target handle to working buffer (the one that gets consumed)
        std::copy(target_handle.pos_taps, target_handle.pos_taps + target_handle.num_pos_taps, target_pos_taps_);
        num_target_pos_taps_ = target_handle.num_pos_taps;
        target_pos_head_ = 0;

        std::copy(target_handle.neg_taps, target_handle.neg_taps + target_handle.num_neg_taps, target_neg_taps_);
        num_target_neg_taps_ = target_handle.num_neg_taps;
        target_neg_head_ = 0;

        // Switch to using current taps for processing
        velvet_pos_taps_ = current_pos_taps_;
        velvet_neg_taps_ = current_neg_taps_;
        
        is_morphing_ = true;
    }

    void MorphIRVelvet_Update()
    {
        if (!is_morphing_)
            return;

        // POSITIVE TAPS - process from end of initial, start of target
        // 1) Substitute positive tap when both initial and target available
        if (num_initial_pos_taps_ > 0 && initial_pos_tail_ < num_initial_pos_taps_ && target_pos_head_ < num_target_pos_taps_)
        {
            size_t new_tap = target_pos_taps_[target_pos_head_];
            SubstitutePositiveTap(initial_pos_tail_, new_tap); 
            if (initial_pos_tail_ == 0) {
                num_initial_pos_taps_ = 0;  // Mark as done to avoid underflow
            } else {
                initial_pos_tail_--;  // Move backwards through initial
            }
            target_pos_head_++;   // Move forwards through target
        }
        // 2) Only remove if no target left  
        else if (num_initial_pos_taps_ > 0 && initial_pos_tail_ < num_initial_pos_taps_)
        {
            RemovePositiveTap();
            if (initial_pos_tail_ == 0) {
                num_initial_pos_taps_ = 0;  // Mark as done to avoid underflow
            } else {
                initial_pos_tail_--;
            }
        }
        // 3) Only add if no initial left
        else if (target_pos_head_ < num_target_pos_taps_)
        {
            size_t tap_to_add = target_pos_taps_[target_pos_head_];
            AddPositiveTap(tap_to_add); 
            target_pos_head_++;
        }

        // NEGATIVE TAPS - process from end of initial, start of target
        // 4) Substitute negative tap when both initial and target available
        if (num_initial_neg_taps_ > 0 && initial_neg_tail_ < num_initial_neg_taps_ && target_neg_head_ < num_target_neg_taps_)
        {
            size_t new_tap = target_neg_taps_[target_neg_head_];
            SubstituteNegativeTap(initial_neg_tail_, new_tap); 
            if (initial_neg_tail_ == 0) {
                num_initial_neg_taps_ = 0;  // Mark as done to avoid underflow
            } else {
                initial_neg_tail_--;  // Move backwards through initial
            }
            target_neg_head_++;   // Move forwards through target
        }
        // 5) Only remove if no target left
        else if (num_initial_neg_taps_ > 0 && initial_neg_tail_ < num_initial_neg_taps_)
        {
            RemoveNegativeTap(); 
            if (initial_neg_tail_ == 0) {
                num_initial_neg_taps_ = 0;  // Mark as done to avoid underflow
            } else {
                initial_neg_tail_--;
            }
        }
        // 6) Only add if no initial left
        else if (target_neg_head_ < num_target_neg_taps_)
        {
            size_t tap_to_add = target_neg_taps_[target_neg_head_];
            AddNegativeTap(tap_to_add); 
            target_neg_head_++;
        }

        // Check if morphing is complete
        bool pos_complete = (num_initial_pos_taps_ == 0) && (target_pos_head_ >= num_target_pos_taps_);
        bool neg_complete = (num_initial_neg_taps_ == 0) && (target_neg_head_ >= num_target_neg_taps_);

        if (pos_complete && neg_complete)
        {
            is_morphing_ = false;
        }
    }

private:
    void RemovePositiveTap()
    {
        assert(num_current_pos_taps_ > 0);
        num_current_pos_taps_--;
        num_velvet_pos_taps_ = num_current_pos_taps_;
    }

    void AddPositiveTap(size_t tap_position)
    {
        assert(num_current_pos_taps_ < max_pos_taps_);
        current_pos_taps_[num_current_pos_taps_] = tap_position;
        num_current_pos_taps_++;
        num_velvet_pos_taps_ = num_current_pos_taps_;
    }

    void RemoveNegativeTap()
    {
        assert(num_current_neg_taps_ > 0);
        num_current_neg_taps_--;
        num_velvet_neg_taps_ = num_current_neg_taps_;
    }

    void AddNegativeTap(size_t tap_position)
    {
        assert(num_current_neg_taps_ < max_neg_taps_);
        current_neg_taps_[num_current_neg_taps_] = tap_position;
        num_current_neg_taps_++;
        num_velvet_neg_taps_ = num_current_neg_taps_;
    }

    void SubstitutePositiveTap(size_t index, size_t new_tap_position)
    {
        assert(index < num_current_pos_taps_);
        current_pos_taps_[index] = new_tap_position;
    }

    void SubstituteNegativeTap(size_t index, size_t new_tap_position)
    {
        assert(index < num_current_neg_taps_);
        current_neg_taps_[index] = new_tap_position;
    }

protected:
    template <WrappingMode WMode, ChannelLayout CLayout>
    void set_process_function(WrappingMode)
    {
        active_process_function_ = &VelvetConvolutionEngine::ProcessImpl<WMode, CLayout>;
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
            // VELVET KERNEL
            for (size_t t = 0; t < num_velvet_pos_taps_; t++)
                ConvolutionUtils::for_each_channel<CLayout>([&](size_t ch) 
                {
                    const size_t tap_pos = ConvolutionUtils::wrap_address<WMode>(write_head_ + ch + velvet_pos_taps_[t] * num_ch, buffer_size_);
                    circ_buffer_[tap_pos] += in[smp*num_ch + ch];
                }, num_channels_);
            for (size_t t = 0; t < num_velvet_neg_taps_; t++)
                ConvolutionUtils::for_each_channel<CLayout>([&](size_t ch) 
                {
                    const size_t tap_pos = ConvolutionUtils::wrap_address<WMode>(write_head_ + ch + velvet_neg_taps_[t] * num_ch, buffer_size_);
                    circ_buffer_[tap_pos] -= in[smp*num_ch + ch];
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

	// Used for VELVET kernel
    const size_t* velvet_pos_taps_;
    size_t        num_velvet_pos_taps_;
    const size_t* velvet_neg_taps_;
    size_t        num_velvet_neg_taps_;

    // Morphing state
    bool is_morphing_;
    
    // Working buffers for current IR state
    size_t* current_pos_taps_;
    size_t* current_neg_taps_;
    size_t num_current_pos_taps_;
    size_t num_current_neg_taps_;
    
    // Working buffers for initial IR (gets consumed during morphing)
    size_t* initial_pos_taps_;
    size_t* initial_neg_taps_;
    size_t num_initial_pos_taps_;
    size_t num_initial_neg_taps_;
    size_t initial_pos_tail_;  // Process from end of initial arrays
    size_t initial_neg_tail_;
    
    // Working buffers for target IR (gets consumed during morphing)
    size_t* target_pos_taps_;
    size_t* target_neg_taps_;
    size_t num_target_pos_taps_;
    size_t num_target_neg_taps_;
    size_t target_pos_head_;  // Process from start of target arrays
    size_t target_neg_head_;
    
    // Buffer size limits
    size_t max_pos_taps_;
    size_t max_neg_taps_;
};

#endif // VELVET_CONVOLUTION_ENGINE_H