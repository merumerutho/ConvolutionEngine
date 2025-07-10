#pragma once
#ifndef IR_HANDLE_H
#define IR_HANDLE_H

#include <cstddef>

/**
 * @brief A handle for a standard, dense impulse response.
 */
struct DenseIRHandle
{
    const float*  taps;          /**< Pointer to the array of IR tap values. */
    size_t        num_taps;      /**< The total number of taps in the IR. */
};

/**
 * @brief A handle for a generic sparse impulse response with explicit tap values.
 */
struct SparseIRHandle
{
    const size_t* positions;     /**< Pointer to the array of tap positions. */
    const float*  values;        /**< Pointer to the array of corresponding tap values. */
    size_t        num_taps;      /**< The number of sparse taps. */
};

struct VelvetIRHandle
{
    const size_t* pos_taps;      /**< Pointer to the array of positive tap positions. */
    size_t        num_pos_taps;  /**< The number of positive taps. */
    const size_t* neg_taps;      /**< Pointer to the array of negative tap positions. */
    size_t        num_neg_taps;  /**< The number of negative taps. */
};


#endif // IR_HANDLE_H
