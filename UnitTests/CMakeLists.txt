# Minimum version of CMake required
cmake_minimum_required(VERSION 3.14)

# Name of your project
project(VelvetConvolutionTestProject)

# Set the C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
# set(CMAKE_BUILD_TYPE Release)

# --- Make libraries static ---
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic")

# --- Include FetchContent to manage dependencies ---
include(FetchContent)

# --- Declare the Google Test dependency ---
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# --- Make the dependency available to your project ---
# This will download and configure gtest, but not build it yet
FetchContent_MakeAvailable(googletest)

# --- Set up your test executable ---

# Enable testing for the project
enable_testing()

# Add your test runner executable, which is built from your test file
file(GLOB_RECURSE TEST_SOURCES "src/*.cpp")

add_executable(
  run_tests
  ${TEST_SOURCES}
)

# Link your test runner against Google Test's libraries.
# CMake makes the 'gtest_main' target available after FetchContent_MakeAvailable.
# This target automatically links against gtest, gtest_main, and pthread.
target_include_directories(
  run_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(
  run_tests
  PRIVATE gtest_main
)

# Discover and add the tests from your executable to CTest
include(GoogleTest)
gtest_discover_tests(run_tests)