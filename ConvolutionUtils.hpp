#pragma once
#ifndef CONVOLUTION_UTILS_H
#define CONVOLUTION_UTILS_H

#include <cstddef>

/**
 * @brief Common utilities shared across all convolution engine types
 */
namespace ConvolutionUtils {

    enum class WrappingMode { POWER_OF_TWO, ARBITRARY };
    enum class ChannelLayout { MONO, STEREO, QUAD, MULTICHANNEL };

    /**
     * @brief Wraps buffer address for circular buffer access
     */
    template <WrappingMode WMode>
    [[nodiscard]] constexpr size_t wrap_address(size_t addr, size_t buffer_size) noexcept
    {
        if constexpr (WMode == WrappingMode::POWER_OF_TWO) return addr & (buffer_size-1);
        else return addr % buffer_size;
    }

    /**
     * @brief Executes a function for each channel based on channel layout
     */
    template<ChannelLayout CLayout, typename Func>
    void for_each_channel(Func&& f, size_t num_channels)
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
            for (size_t k = 0; k < num_channels; k++)
            {
                f(k);
            }
        }
    }

    /**
     * @brief Checks if a number is a power of two
     */
    inline bool is_power_of_two(size_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

} // namespace ConvolutionUtils

#endif // CONVOLUTION_UTILS_H