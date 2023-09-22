
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation"
        "c++")

register_flag_optional(CMAKE_CUDA_COMPILER "Path to the CUDA nvcc compiler" "")

# XXX we may want to drop this eventually and use CMAKE_CUDA_ARCHITECTURES directly
register_flag_required(DEVICE_ARCH "Nvidia architecture, will be passed in via `-arch=` (e.g `sm_70`) for nvcc")

register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH`"
        "")

register_flag_optional(MANAGED_ALLOC "Use UVM (cudaMallocManaged) instead of the device-only allocation (cudaMalloc)"
        "OFF")

register_flag_optional(SYNC_ALL_KERNELS
        "Fully synchronise all kernels after launch, this also enables synchronous error checking with line and file name"
        "OFF")

register_flag_required(RAJA_BACK_END "Specify whether we target HIP or RAJA")


macro(setup)
    if (POLICY CMP0104)
        cmake_policy(SET CMP0104 OLD)
    endif ()

    find_package(Threads REQUIRED)
    # Add umpire to manage resource
    find_package(UMPIRE REQUIRED)
    register_link_library(umpire)
    find_package(raja REQUIRED)
    register_link_library(RAJA)

    if (${RAJA_BACK_END} STREQUAL "CUDA")
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CXX_STANDARD 17)
      enable_language(CUDA)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -arch=${DEVICE_ARCH} -extended-lambda -restrict -keep ${CUDA_EXTRA_FLAGS}")
      # -use_fast_math
      # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
      # appended later
     wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})
      register_definitions(__ENABLE_CUDA__)
    elseif (${RAJA_BACK_END} STREQUAL "HIP")
      set(CMAKE_CXX_STANDARD 17)
      find_package(hip REQUIRED)
      register_definitions(__ENABLE_HIP__)
    else()
      message(FATAL_ERROR "Variable RAJA_BACK_END needs to be set to either HIP or CUDA")
    endif()
    if (MANAGED_ALLOC)
        register_definitions(CLOVER_MANAGED_ALLOC)
    endif ()

    if (SYNC_ALL_KERNELS)
        register_definitions(CLOVER_SYNC_ALL_KERNELS)
    endif ()

    message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
endmacro()

macro(setup_target NAME)
    # Treat everything as CUDA source
    get_target_property(PROJECT_SRC "${NAME}" SOURCES)
  if (${RAJA_BACK_END} STREQUAL "CUDA")
    foreach (SRC ${PROJECT_SRC})
        set_source_files_properties("${SRC}" PROPERTIES LANGUAGE CUDA)
    endforeach ()
    set_property(TARGET ${NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  elseif (${RAJA_BACK_END} STREQUAL "HIP")
      set_property(TARGET ${NAME} PROPERTY HIP_ARCHITECTURES ${DEVICE_ARCH})
  endif()
endmacro()
