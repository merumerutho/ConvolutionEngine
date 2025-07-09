#pragma once
#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include "IRHandle.h"
#include <cstddef>   
#include <algorithm> 
#include <cmath>     
#include <type_traits> 

/**
 * @brief A generic, real-time safe convolution engine supporting dense, sparse,
 * and velvet impulse responses with compile-time optimization for channel layout.
 */
class ConvolutionEngine
{
    friend class ConvolutionEngineTest;
public:
    enum class IRType { DENSE, SPARSE, VELVET };
    enum class WrappingMode { POWER_OF_TWO, ARBITRARY };
    enum class ChannelLayout { MONO, STEREO, QUAD, MULTICHANNEL };

    using ProcessFunctionPtr = void (ConvolutionEngine::*)(const float*, float*, size_t);

    ConvolutionEngine() : active_process_function_(nullptr) {}
    ~ConvolutionEngine() {}

    template <typename IRHandleType>
    void Init(const IRHandleType& handle, float* circ_buffer, size_t buffer_size)
    {
        circ_buffer_  = circ_buffer;
        buffer_size_  = buffer_size;
        num_channels_ = handle.num_channels;
        write_head_   = 0;

        assert(circ_buffer_); 	// != NULL
        assert(buffer_size);  	// > 0 
        assert(num_channels_); 	// > 0

		// Init empty buffer
        std::fill(circ_buffer_, circ_buffer_ + buffer_size_, 0.0f);

        const bool is_pow2 = is_power_of_two(buffer_size_);
        const auto wrapping_mode = is_pow2 ? WrappingMode::POWER_OF_TWO : WrappingMode::ARBITRARY;

        IRType ir_type;
        if constexpr (std::is_same_v<IRHandleType, DenseIRHandle>)
        {
            ir_type = IRType::DENSE;
            dense_taps_ 				= handle.taps;
            num_dense_taps_ 			= handle.num_taps;
            assert(dense_taps_          || (num_dense_taps_ == 0));         // != NULL
        }
        else if constexpr (std::is_same_v<IRHandleType, SparseIRHandle>)
        {
            ir_type = IRType::SPARSE;
            sparse_positions_ 			= handle.positions;
            sparse_values_    			= handle.values;
            num_sparse_taps_  			= handle.num_taps;
            assert(sparse_positions_    || (num_sparse_taps_ == 0));        // != NULL
            assert(sparse_values_       || (num_sparse_taps_ == 0));        // != NULL
        }
        else if constexpr (std::is_same_v<IRHandleType, VelvetIRHandle>)
        {
            ir_type = IRType::VELVET;
            velvet_pos_taps_ 			= handle.pos_taps;
            num_velvet_pos_taps_ 		= handle.num_pos_taps;
            velvet_neg_taps_ 			= handle.neg_taps;
            num_velvet_neg_taps_ 		= handle.num_neg_taps;
            assert(velvet_pos_taps_     || (num_velvet_pos_taps_ == 0));    // != NULL
            assert(velvet_neg_taps_     || (num_velvet_neg_taps_ == 0));    // != NULL
        }

        // Dispatch to the correct template specialization based on runtime channel count
        // and IRType/WrappingMode
        if (num_channels_ == 1)
		{
            dispatch_set_process_function(ir_type, wrapping_mode, ChannelLayout::MONO);
		}
        else if (num_channels_ == 2)
		{
            dispatch_set_process_function(ir_type, wrapping_mode, ChannelLayout::STEREO);
		}
        else if (num_channels_ == 4)
		{
            dispatch_set_process_function(ir_type, wrapping_mode, ChannelLayout::QUAD);
		}
        else
		{
            dispatch_set_process_function(ir_type, wrapping_mode, ChannelLayout::MULTICHANNEL);
		}
    }

    void Process(const float* in, float* out, size_t size)
    {
        (this->*active_process_function_)(in, out, size);
    }

protected:
    template <IRType IType, WrappingMode WMode, ChannelLayout CLayout>
    void set_process_function(IRType, WrappingMode)
    {
        active_process_function_ = &ConvolutionEngine::ProcessImpl<IType, WMode, CLayout>;
    }

    void dispatch_set_process_function(IRType ir_type, WrappingMode wrapping_mode, ChannelLayout channel_layout)
    {
		// DENSE
        if (ir_type == IRType::DENSE) {
			// POWER OF TWO
            if (wrapping_mode == WrappingMode::POWER_OF_TWO) {
                if (channel_layout == ChannelLayout::MONO) set_process_function<IRType::DENSE, WrappingMode::POWER_OF_TWO, ChannelLayout::MONO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::STEREO) set_process_function<IRType::DENSE, WrappingMode::POWER_OF_TWO, ChannelLayout::STEREO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::QUAD) set_process_function<IRType::DENSE, WrappingMode::POWER_OF_TWO, ChannelLayout::QUAD>(ir_type, wrapping_mode);
                else set_process_function<IRType::DENSE, WrappingMode::POWER_OF_TWO, ChannelLayout::MULTICHANNEL>(ir_type, wrapping_mode);
            } else { // ARBITRARY
                if (channel_layout == ChannelLayout::MONO) set_process_function<IRType::DENSE, WrappingMode::ARBITRARY, ChannelLayout::MONO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::STEREO) set_process_function<IRType::DENSE, WrappingMode::ARBITRARY, ChannelLayout::STEREO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::QUAD) set_process_function<IRType::DENSE, WrappingMode::ARBITRARY, ChannelLayout::QUAD>(ir_type, wrapping_mode);
                else set_process_function<IRType::DENSE, WrappingMode::ARBITRARY, ChannelLayout::MULTICHANNEL>(ir_type, wrapping_mode);
            }
		// SPARSE
        } else if (ir_type == IRType::SPARSE) {
			// POWER OF TWO
            if (wrapping_mode == WrappingMode::POWER_OF_TWO) {
                if (channel_layout == ChannelLayout::MONO) set_process_function<IRType::SPARSE, WrappingMode::POWER_OF_TWO, ChannelLayout::MONO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::STEREO) set_process_function<IRType::SPARSE, WrappingMode::POWER_OF_TWO, ChannelLayout::STEREO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::QUAD) set_process_function<IRType::SPARSE, WrappingMode::POWER_OF_TWO, ChannelLayout::QUAD>(ir_type, wrapping_mode);
                else set_process_function<IRType::SPARSE, WrappingMode::POWER_OF_TWO, ChannelLayout::MULTICHANNEL>(ir_type, wrapping_mode);
            } else { // ARBITRARY
                if (channel_layout == ChannelLayout::MONO) set_process_function<IRType::SPARSE, WrappingMode::ARBITRARY, ChannelLayout::MONO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::STEREO) set_process_function<IRType::SPARSE, WrappingMode::ARBITRARY, ChannelLayout::STEREO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::QUAD) set_process_function<IRType::SPARSE, WrappingMode::ARBITRARY, ChannelLayout::QUAD>(ir_type, wrapping_mode);
                else set_process_function<IRType::SPARSE, WrappingMode::ARBITRARY, ChannelLayout::MULTICHANNEL>(ir_type, wrapping_mode);
            }
		// VELVET
        } else { 
			// POWER OF TWO
            if (wrapping_mode == WrappingMode::POWER_OF_TWO) {	
                if (channel_layout == ChannelLayout::MONO) set_process_function<IRType::VELVET, WrappingMode::POWER_OF_TWO, ChannelLayout::MONO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::STEREO) set_process_function<IRType::VELVET, WrappingMode::POWER_OF_TWO, ChannelLayout::STEREO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::QUAD) set_process_function<IRType::VELVET, WrappingMode::POWER_OF_TWO, ChannelLayout::QUAD>(ir_type, wrapping_mode);
                else set_process_function<IRType::VELVET, WrappingMode::POWER_OF_TWO, ChannelLayout::MULTICHANNEL>(ir_type, wrapping_mode);
            } else { // ARBITRARY
                if (channel_layout == ChannelLayout::MONO) set_process_function<IRType::VELVET, WrappingMode::ARBITRARY, ChannelLayout::MONO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::STEREO) set_process_function<IRType::VELVET, WrappingMode::ARBITRARY, ChannelLayout::STEREO>(ir_type, wrapping_mode);
                else if (channel_layout == ChannelLayout::QUAD) set_process_function<IRType::VELVET, WrappingMode::ARBITRARY, ChannelLayout::QUAD>(ir_type, wrapping_mode);
                else set_process_function<IRType::VELVET, WrappingMode::ARBITRARY, ChannelLayout::MULTICHANNEL>(ir_type, wrapping_mode);
            }
        }
    }

    template <WrappingMode WMode>
    [[nodiscard]] constexpr size_t wrap_address(size_t addr) const noexcept
    {
        if constexpr (WMode == WrappingMode::POWER_OF_TWO) return addr & (buffer_size_-1);
        else return addr % buffer_size_;
    }

    template<ChannelLayout CLayout, typename Func>
    void for_each_channel(Func&& f)
    {
        if constexpr (CLayout == ChannelLayout::MONO)
        {
            f(0);
        }
        else if constexpr (CLayout == ChannelLayout::STEREO)
        {
            f(0);
            f(1);
        }
		else if constexpr (CLayout == ChannelLayout::QUAD)
        {
            f(0);
            f(1);
            f(2);
            f(3);
        }
        else // MULTICHANNEL
        {
            for (size_t k = 0; k < num_channels_; k++)
            {
                f(k);
            }
        }
    }

    template <IRType IType, WrappingMode WMode, ChannelLayout CLayout>
    void ProcessImpl(const float* in, float* out, size_t size)
    {
        const size_t num_ch = num_channels_;

        for (size_t i = 0; i < size; i++)
        {
            // DENSE KERNEL
            if constexpr (IType == IRType::DENSE)
            {
                for (size_t j = 0; j < num_dense_taps_; j++)
                    for_each_channel<CLayout>([&](size_t ch) 
                    {
                        const size_t tap_pos = wrap_address<WMode>(write_head_ + ch + j * num_ch);
                        circ_buffer_[tap_pos] += in[i*num_ch + ch] * dense_taps_[j];
                    });
            }
            // SPARSE KERNEL
            else if constexpr (IType == IRType::SPARSE)
            {
                for (size_t j = 0; j < num_sparse_taps_; j++)
                    for_each_channel<CLayout>([&](size_t ch) 
                    {
                        const size_t tap_pos = wrap_address<WMode>(write_head_ + ch + sparse_positions_[j] * num_ch);
                        circ_buffer_[tap_pos] += in[i*num_ch + ch] * sparse_values_[j];
                    });
            }
            // VELVET KERNEL
            else if constexpr (IType == IRType::VELVET)
            {
                for (size_t j = 0; j < num_velvet_pos_taps_; j++)
                    for_each_channel<CLayout>([&](size_t ch) 
                    {
                        const size_t tap_pos = wrap_address<WMode>(write_head_ + ch + velvet_pos_taps_[j] * num_ch);
                        circ_buffer_[tap_pos] += in[i*num_ch + ch];
                    });
                for (size_t j = 0; j < num_velvet_neg_taps_; j++)
                    for_each_channel<CLayout>([&](size_t ch) 
                    {
                        const size_t tap_pos = wrap_address<WMode>(write_head_ + ch + velvet_neg_taps_[j] * num_ch);
                        circ_buffer_[tap_pos] -= in[i*num_ch + ch];
                    });
            }
            
            // HEAD ADVANCE
            for_each_channel<CLayout>([&](size_t ch) {
                out[i*num_ch + ch] = circ_buffer_[write_head_];
                circ_buffer_[write_head_] = 0.0f;
				write_head_ = wrap_address<WMode>(write_head_ + 1);
            });
        }
    }

    inline bool is_power_of_two(size_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

    ProcessFunctionPtr active_process_function_;

    size_t write_head_;
    size_t buffer_size_;
    size_t num_channels_;
    float* circ_buffer_;

	// Used for DENSE kernel
    const float * dense_taps_;
    size_t        num_dense_taps_;
	
	// Used for SPARSE kernel
    const size_t* sparse_positions_;
    const float*  sparse_values_;
    size_t        num_sparse_taps_;
	
	// Used for VELVET kernel
    const size_t* velvet_pos_taps_;
    size_t        num_velvet_pos_taps_;
    const size_t* velvet_neg_taps_;
    size_t        num_velvet_neg_taps_;
};

#endif // CONVOLUTION_ENGINE_H