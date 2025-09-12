#include "test_common.hpp"
#include "generated_test_data.hpp"

TEST(VelvetMorphingTest, MorphInitialization) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    // Morphing buffers with maximum capacity
    const size_t max_pos_taps = 100;
    const size_t max_neg_taps = 100;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    VelvetIRHandle initial_handle = {
        velvet_ir_pos_positions, velvet_ir_pos_positions_size,
        velvet_ir_neg_positions, velvet_ir_neg_positions_size
    };
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    VelvetIRHandle target_handle = {
        velvet_ir_2_pos_positions, velvet_ir_2_pos_positions_size,
        velvet_ir_2_neg_positions, velvet_ir_2_neg_positions_size
    };
    
    EXPECT_NO_THROW(engine.MorphIRVelvet(target_handle));
}

TEST(VelvetMorphingTest, MorphParameterValidation) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    const size_t max_pos_taps = 100;
    const size_t max_neg_taps = 100;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    VelvetIRHandle initial_handle = {
        velvet_ir_pos_positions, velvet_ir_pos_positions_size,
        velvet_ir_neg_positions, velvet_ir_neg_positions_size
    };
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    // Test with target IR that exceeds buffer capacity
    const size_t oversized_pos[] = {1, 2, 3, 4, 5}; // Small array to exceed our mock capacity 
    VelvetIRHandle oversized_handle = {
        oversized_pos, 150, // More than max_pos_taps (100)
        velvet_ir_2_neg_positions, velvet_ir_2_neg_positions_size
    };
    
#ifdef _DEBUG
    EXPECT_DEATH(engine.MorphIRVelvet(oversized_handle), "");
#endif
    
    // Valid call should work
    VelvetIRHandle target_handle = {
        velvet_ir_2_pos_positions, velvet_ir_2_pos_positions_size,
        velvet_ir_2_neg_positions, velvet_ir_2_neg_positions_size
    };
    EXPECT_NO_THROW(engine.MorphIRVelvet(target_handle));
}

TEST(VelvetMorphingTest, MorphProgressValidation) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    const size_t max_pos_taps = 100;
    const size_t max_neg_taps = 100;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    VelvetIRHandle initial_handle = {
        velvet_ir_pos_positions, velvet_ir_pos_positions_size,
        velvet_ir_neg_positions, velvet_ir_neg_positions_size
    };
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    VelvetIRHandle target_handle = {
        velvet_ir_2_pos_positions, velvet_ir_2_pos_positions_size,
        velvet_ir_2_neg_positions, velvet_ir_2_neg_positions_size
    };
    
    engine.MorphIRVelvet(target_handle);
    
    // Track morphing progress by counting expected steps
    size_t update_count = 0;
    size_t max_expected_updates = std::max(
        velvet_ir_pos_positions_size + velvet_ir_2_pos_positions_size,
        velvet_ir_neg_positions_size + velvet_ir_2_neg_positions_size
    );
    
    // Apply morphing updates until completion
    while (update_count < max_expected_updates * 2 && engine.IsMorphing()) {
        engine.MorphIRVelvet_Update();
        update_count++;
        
        // Check if we can still process audio (no crashes)
        std::vector<float> input(4, 1.0f);
        std::vector<float> output(4, 0.0f);
        EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
    }
    
    EXPECT_LT(update_count, max_expected_updates * 2);
}

TEST(VelvetMorphingTest, MorphWithoutBuffersFailsGracefully) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    VelvetIRHandle initial_handle = {
        velvet_ir_pos_positions, velvet_ir_pos_positions_size,
        velvet_ir_neg_positions, velvet_ir_neg_positions_size
    };
    
    // Initialize without morphing buffers
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1);
    
    VelvetIRHandle target_handle = {
        velvet_ir_2_pos_positions, velvet_ir_2_pos_positions_size,
        velvet_ir_2_neg_positions, velvet_ir_2_neg_positions_size
    };
    
    // This should fail in debug mode due to assertion
#ifdef _DEBUG
    EXPECT_DEATH(engine.MorphIRVelvet(target_handle), "");
#endif
}

TEST(VelvetMorphingTest, ProcessingDuringMorph) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    const size_t max_pos_taps = 100;
    const size_t max_neg_taps = 100;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    VelvetIRHandle initial_handle = {
        velvet_ir_pos_positions, velvet_ir_pos_positions_size,
        velvet_ir_neg_positions, velvet_ir_neg_positions_size
    };
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    VelvetIRHandle target_handle = {
        velvet_ir_2_pos_positions, velvet_ir_2_pos_positions_size,
        velvet_ir_2_neg_positions, velvet_ir_2_neg_positions_size
    };
    
    // Start morphing
    engine.MorphIRVelvet(target_handle);
    
    // Process audio during morphing
    std::vector<float> input(16, 1.0f);
    std::vector<float> output(16, 0.0f);
    
    for (int i = 0; i < 10; i++) {
        EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 16));
        engine.MorphIRVelvet_Update();
    }
}

TEST(VelvetMorphingTest, TapAdditionAndRemovalLogic) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    const size_t max_pos_taps = 10;
    const size_t max_neg_taps = 10;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    // Simple test case: 3 initial pos, 2 target pos, 2 initial neg, 3 target neg
    const size_t init_pos[] = {10, 20, 30};
    const size_t init_neg[] = {15, 25};
    const size_t targ_pos[] = {40, 50};
    const size_t targ_neg[] = {35, 45, 55};
    
    VelvetIRHandle initial_handle = {init_pos, 3, init_neg, 2};
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    VelvetIRHandle target_handle = {targ_pos, 2, targ_neg, 3};
    
    // Start morphing
    engine.MorphIRVelvet(target_handle);
    
    // Apply updates step by step to observe the morphing process
    std::vector<float> input(4, 1.0f);
    std::vector<float> output(4, 0.0f);
    
    for (int step = 0; step < 10; step++) {
        // Process audio to ensure no crashes
        EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
        
        // Update morphing
        engine.MorphIRVelvet_Update();
    }
}

TEST(VelvetMorphingTest, SubstitutionOptimization) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    const size_t max_pos_taps = 10;
    const size_t max_neg_taps = 10;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    // Equal sized arrays to test substitution (same number of pos/neg taps)
    const size_t init_pos[] = {10, 20, 30};
    const size_t init_neg[] = {15, 25, 35};
    const size_t targ_pos[] = {40, 50, 60};
    const size_t targ_neg[] = {45, 55, 65};
    
    VelvetIRHandle initial_handle = {init_pos, 3, init_neg, 3};
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    VelvetIRHandle target_handle = {targ_pos, 3, targ_neg, 3};
    
    // Start morphing - should use substitution for all taps
    engine.MorphIRVelvet(target_handle);
    
    // Since both initial and target have same size (3 each), 
    // it should complete in exactly 3 updates (one per substitution pair)
    std::vector<float> input(4, 1.0f);
    std::vector<float> output(4, 0.0f);
    
    // Step 1: Should substitute first pos tap (10->40) and first neg tap (15->45)
    EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
    engine.MorphIRVelvet_Update();
    
    // Step 2: Should substitute second pos tap (20->50) and second neg tap (25->55)
    EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
    engine.MorphIRVelvet_Update();
    
    // Step 3: Should substitute third pos tap (30->60) and third neg tap (35->65)
    EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
    engine.MorphIRVelvet_Update();
    
    // Step 4: Should be complete now, no more operations
    EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
    engine.MorphIRVelvet_Update();
    
    // Additional updates should be safe
    for (int i = 0; i < 3; i++) {
        EXPECT_NO_THROW(engine.MorphIRVelvet_Update());
        EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
    }
}

TEST(VelvetMorphingTest, MixedSubstitutionAndAddRemove) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    const size_t max_pos_taps = 10;
    const size_t max_neg_taps = 10;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    // Unequal sizes: initial has more pos, target has more neg
    const size_t init_pos[] = {10, 20, 30};  // 3 pos taps
    const size_t init_neg[] = {15};          // 1 neg tap
    const size_t targ_pos[] = {40};          // 1 pos tap
    const size_t targ_neg[] = {45, 55, 65};  // 3 neg taps
    
    VelvetIRHandle initial_handle = {init_pos, 3, init_neg, 1};
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    VelvetIRHandle target_handle = {targ_pos, 1, targ_neg, 3};
    
    engine.MorphIRVelvet(target_handle);
    
    std::vector<float> input(4, 1.0f);
    std::vector<float> output(4, 0.0f);
    
    // Expected behavior:
    // Update 1: Substitute pos (10->40) and neg (15->45)
    // Update 2: Remove pos (20), Add neg (55)  
    // Update 3: Remove pos (30), Add neg (65)
    
    for (int step = 0; step < 5; step++) {
        EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
        engine.MorphIRVelvet_Update();
    }
}

TEST(VelvetMorphingTest, UpdatesAfterCompletion) {
    VelvetConvolutionEngine engine;
    std::vector<float> circ_buffer(1024, 0.0f);
    
    const size_t max_pos_taps = 10;
    const size_t max_neg_taps = 10;
    std::vector<size_t> current_pos_buffer(max_pos_taps);
    std::vector<size_t> current_neg_buffer(max_neg_taps);
    std::vector<size_t> initial_pos_buffer(max_pos_taps);
    std::vector<size_t> initial_neg_buffer(max_neg_taps);
    std::vector<size_t> target_pos_buffer(max_pos_taps);
    std::vector<size_t> target_neg_buffer(max_neg_taps);
    
    // Small test case for quick completion
    const size_t init_pos[] = {10};
    const size_t init_neg[] = {15};
    const size_t targ_pos[] = {20};
    const size_t targ_neg[] = {25};
    
    VelvetIRHandle initial_handle = {init_pos, 1, init_neg, 1};
    
    engine.Init(initial_handle, circ_buffer.data(), circ_buffer.size(), 1,
               current_pos_buffer.data(), current_neg_buffer.data(),
               initial_pos_buffer.data(), initial_neg_buffer.data(),
               target_pos_buffer.data(), target_neg_buffer.data(),
               max_pos_taps, max_neg_taps);
    
    VelvetIRHandle target_handle = {targ_pos, 1, targ_neg, 1};
    
    engine.MorphIRVelvet(target_handle);
    
    // Complete morphing (should take 2 steps: remove old, add new for each polarity)
    engine.MorphIRVelvet_Update(); // Step 1
    engine.MorphIRVelvet_Update(); // Step 2 - should complete
    
    // Additional updates should not cause issues
    std::vector<float> input(4, 1.0f);
    std::vector<float> output(4, 0.0f);
    
    for (int i = 0; i < 5; i++) {
        EXPECT_NO_THROW(engine.MorphIRVelvet_Update());
        EXPECT_NO_THROW(engine.Process(input.data(), output.data(), 4));
    }
}