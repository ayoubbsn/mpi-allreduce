cmake_minimum_required(VERSION 3.12)
project(mpi_allreduce LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS C)

add_compile_options(-O3 -Wall -Wextra)

add_library(allreduce STATIC
  src/allreduce_tree.c
  src/allreduce_ring.c
  src/common.c)
target_include_directories(allreduce PUBLIC include)
target_link_libraries(allreduce PUBLIC MPI::MPI_C)

add_executable(bench src/bench.c)
target_link_libraries(bench PRIVATE allreduce m)

add_executable(test_correctness tests/test_correctness.c)
target_link_libraries(test_correctness PRIVATE allreduce m)

# Tests
enable_testing()

set(TEST_NPROCS "1;2;3;4;5;8" CACHE STRING "Process counts used by ctest")

# Open MPI refuses to start more ranks than cores (and to run as root)
# unless told otherwise; CI runners often have 2 cores.
set(MPIEXEC_EXTRA_FLAGS "")
execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version
                OUTPUT_VARIABLE MPIEXEC_VERSION_OUTPUT
                ERROR_VARIABLE MPIEXEC_VERSION_OUTPUT
                RESULT_VARIABLE MPIEXEC_VERSION_RESULT)
if(MPIEXEC_VERSION_OUTPUT MATCHES "Open MPI|OpenRTE")
  message(STATUS "Open MPI detected: tests run with --oversubscribe")
  list(APPEND MPIEXEC_EXTRA_FLAGS --oversubscribe)
endif()

foreach(np IN LISTS TEST_NPROCS)
  add_test(NAME correctness_np${np}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
            ${MPIEXEC_EXTRA_FLAGS} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_correctness> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(correctness_np${np} PROPERTIES
    PROCESSORS ${np}
    TIMEOUT 300
    FAIL_REGULAR_EXPRESSION "FAIL")
endforeach()

# Short benchmark run: checks that it starts, runs and writes a CSV.
add_test(NAME bench_smoke
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
          ${MPIEXEC_EXTRA_FLAGS} ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:bench> -m 64K -r 1 ${MPIEXEC_POSTFLAGS})
set_tests_properties(bench_smoke PROPERTIES
  PROCESSORS 4
  TIMEOUT 120
  PASS_REGULAR_EXPRESSION "65536,[0-9]+,")
