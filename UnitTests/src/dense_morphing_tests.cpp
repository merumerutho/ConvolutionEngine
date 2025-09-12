#include "test_common.hpp"
#include "generated_test_data.hpp"

TEST(DenseMorphingTest, MorphInitialization) {
    DenseConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    std::vector<float> current_taps(dense_ir_size, 0.0f);
    std::vector<float> morph_delta(dense_ir_size, 0.0f);
    
    DenseIRHandle initial_handle = {dense_ir, dense_ir_size};
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_taps.data(), morph_delta.data());
    
    DenseIRHandle target_handle = {dense_ir_2, dense_ir_2_size};
    
    EXPECT_NO_THROW(engine.MorphIRDense(target_handle, 10));
}

TEST(DenseMorphingTest, MorphParameterValidation) {
    DenseConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    std::vector<float> current_taps(dense_ir_size, 0.0f);
    std::vector<float> morph_delta(dense_ir_size, 0.0f);
    
    DenseIRHandle initial_handle = {dense_ir, dense_ir_size};
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_taps.data(), morph_delta.data());
    
    DenseIRHandle target_handle = {dense_ir_2, dense_ir_2_size};
    
    // Test invalid cycles parameter - this should trigger assertion in debug mode
    // In release mode, we can't directly test assertions, so we skip this test
#ifdef _DEBUG
    EXPECT_DEATH(engine.MorphIRDense(target_handle, 0), "");
    EXPECT_DEATH(engine.MorphIRDense(target_handle, -1), "");
#endif
    
    // Valid call should work
    EXPECT_NO_THROW(engine.MorphIRDense(target_handle, 5));
}

TEST(DenseMorphingTest, MorphProgressValidation) {
    DenseConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    std::vector<float> current_taps(dense_ir_size, 0.0f);
    std::vector<float> morph_delta(dense_ir_size, 0.0f);
    
    DenseIRHandle initial_handle = {dense_ir, dense_ir_size};
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_taps.data(), morph_delta.data());
    
    DenseIRHandle target_handle = {dense_ir_2, dense_ir_2_size};
    
    int morph_cycles = 4;
    engine.MorphIRDense(target_handle, morph_cycles);
    
    // Store initial values for comparison
    std::vector<float> initial_current_taps = current_taps;
    
    // Apply morphing updates and verify progression
    for (int cycle = 0; cycle < morph_cycles; cycle++) {
        engine.MorphIRDense_Update();
        
        // Verify that values are changing towards target
        for (size_t i = 0; i < dense_ir_size; i++) {
            float expected_progress = (cycle + 1) / static_cast<float>(morph_cycles);
            float expected_value = dense_ir[i] + expected_progress * (dense_ir_2[i] - dense_ir[i]);
            
            // Allow small floating point tolerance
            EXPECT_NEAR(current_taps[i], expected_value, 1e-6) 
                << "Tap " << i << " at cycle " << cycle;
        }
    }
    
    // After morphing completion, values should exactly match target
    for (size_t i = 0; i < dense_ir_size; i++) {
        EXPECT_FLOAT_EQ(current_taps[i], dense_ir_2[i]) 
            << "Final tap " << i << " should match target";
    }
}

TEST(DenseMorphingTest, MorphWithoutBuffersFailsGracefully) {
    DenseConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    DenseIRHandle initial_handle = {dense_ir, dense_ir_size};
    // Initialize without morphing buffers
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1);
    
    DenseIRHandle target_handle = {dense_ir_2, dense_ir_2_size};
    
    // This should fail in debug mode due to assertion
#ifdef _DEBUG
    EXPECT_DEATH(engine.MorphIRDense(target_handle, 5), "");
#endif
}

TEST(DenseMorphingTest, MultipleUpdatesAfterCompletion) {
    DenseConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    std::vector<float> current_taps(dense_ir_size, 0.0f);
    std::vector<float> morph_delta(dense_ir_size, 0.0f);
    
    DenseIRHandle initial_handle = {dense_ir, dense_ir_size};
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_taps.data(), morph_delta.data());
    
    DenseIRHandle target_handle = {dense_ir_2, dense_ir_2_size};
    
    // Start morphing with 2 cycles
    engine.MorphIRDense(target_handle, 2);
    
    // Complete the morph
    engine.MorphIRDense_Update(); // Cycle 1
    engine.MorphIRDense_Update(); // Cycle 2 - morphing should complete
    
    // Store completed values
    std::vector<float> completed_values = current_taps;
    
    // Additional updates should not change values
    engine.MorphIRDense_Update();
    engine.MorphIRDense_Update();
    
    for (size_t i = 0; i < dense_ir_size; i++) {
        EXPECT_FLOAT_EQ(current_taps[i], completed_values[i])
            << "Tap " << i << " should not change after completion";
    }
}

TEST(DenseMorphingTest, ProcessingDuringMorph) {
    DenseConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    std::vector<float> current_taps(dense_ir_size, 0.0f);
    std::vector<float> morph_delta(dense_ir_size, 0.0f);
    
    DenseIRHandle initial_handle = {dense_ir, dense_ir_size};
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_taps.data(), morph_delta.data());
    
    DenseIRHandle target_handle = {dense_ir_2, dense_ir_2_size};
    
    // Start morphing
    engine.MorphIRDense(target_handle, 10);
    
    // Process some audio during morphing
    std::vector<float> input(32, 1.0f);  // Impulse-like input
    std::vector<float> output(32, 0.0f);
    
    EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 32));
    
    // Update morph state
    engine.MorphIRDense_Update();
    
    // Process again
    std::fill(output.begin(), output.end(), 0.0f);
    EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 32));
}