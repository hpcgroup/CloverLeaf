register_flag_optional(RAJA_BACK_END "Specify whether we target CPU/CUDA/HIP/SYCL" "CPU")

register_flag_optional(MANAGED_ALLOC "Use UVM (cudaMallocManaged) instead of the device-only allocation (cudaMalloc)"
        "OFF")

register_flag_optional(SYNC_ALL_KERNELS
        "Fully synchronise all kernels after launch, this also enables synchronous error checking with line and file name"
        "OFF")

register_flag_optional(RAJA_IN_TREE "" "")

register_flag_optional(UMPIRE_IN_TREE "" "")

register_flag_optional(RAJA_IN_PACKAGE
        "Use if RAJA is part of a package dependency:
        Path to installation" "")

register_flag_optional(UMPIRE_IN_PACKAGE
        "Use if RAJA is part of a package dependency:
        Path to installation" "")

macro(setup)
    if (POLICY CMP0104)
        cmake_policy(SET CMP0104 OLD)
    endif ()

    set(CMAKE_CXX_STANDARD 17)

    if (EXISTS "${RAJA_IN_TREE}")
        message(STATUS "Building using in-tree RAJA source at `${RAJA_IN_TREE}`")

        # don't build anything that isn't the RAJA library itself, by default their cmake def builds everything, whyyy?
        set(ENABLE_TESTS OFF CACHE BOOL "")
        set(ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(ENABLE_REPRODUCERS OFF CACHE BOOL "")
        set(ENABLE_EXERCISES OFF CACHE BOOL "")
        set(ENABLE_DOCUMENTATION OFF CACHE BOOL "")
        set(ENABLE_BENCHMARKS OFF CACHE BOOL "")

        # RAJA >= v2022.03.0 switched to prefixed variables, we keep the legacy ones for backwards compatibiity
        set(RAJA_ENABLE_TESTS OFF CACHE BOOL "")
        set(RAJA_ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(RAJA_ENABLE_REPRODUCERS OFF CACHE BOOL "")
        set(RAJA_ENABLE_EXERCISES OFF CACHE BOOL "")
        set(RAJA_ENABLE_DOCUMENTATION OFF CACHE BOOL "")
        set(RAJA_ENABLE_BENCHMARKS OFF CACHE BOOL "")

        if (${RAJA_BACK_END} STREQUAL "CUDA")
            set(ENABLE_CUDA ON CACHE BOOL "")
            set(RAJA_ENABLE_CUDA ON CACHE BOOL "")
        endif ()
        
        add_subdirectory(${RAJA_IN_TREE} ${CMAKE_BINARY_DIR}/raja)
    elseif (EXISTS "${RAJA_IN_PACKAGE}")
        message(STATUS "Building using packaged RAJA at `${RAJA_IN_PACKAGE}`")
        find_package(RAJA REQUIRED)
    else ()
        message(FATAL_ERROR "Neither `${RAJA_IN_TREE}` or `${RAJA_IN_PACKAGE}` exists")
    endif ()

    if (EXISTS "${UMPIRE_IN_TREE}")
        message(STATUS "Building using in-tree Umpire source at `${UMPIRE_IN_TREE}`")
        
        if (${RAJA_BACK_END} STREQUAL "CUDA")
            set(UMPIRE_ENABLE_CUDA OFF CACHE BOOL "")
        endif ()
        
        add_subdirectory(${UMPIRE_IN_TREE} ${CMAKE_BINARY_DIR}/umpire)
    elseif (EXISTS "${UMPIRE_IN_PACKAGE}")
        message(STATUS "Building using packaged Umpire at `${UMPIRE_IN_PACKAGE}`")
        find_package(UMPIRE REQUIRED)
    else ()
        message(FATAL_ERROR "Neither `${UMPIRE_IN_TREE}` or `${UMPIRE_IN_PACKAGE}` exists")
    endif ()

    register_link_library(RAJA umpire)
    
    if (${RAJA_BACK_END} STREQUAL "CUDA")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -extended-lambda --expt-relaxed-constexpr --restrict --keep")

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CUDA)
    elseif (${RAJA_BACK_END} STREQUAL "HIP")
        find_package(hip REQUIRED)
    elseif (${RAJA_BACK_END} STREQUAL "SYCL")
        register_definitions(RAJA_TARGET_GPU)
    else()
        register_definitions(RAJA_TARGET_CPU)
        message(STATUS "Falling Back to CPU")
    endif ()   
    
    if (MANAGED_ALLOC)
        register_definitions(CLOVER_MANAGED_ALLOC)
    endif ()

    if (SYNC_ALL_KERNELS)
        register_definitions(CLOVER_SYNC_ALL_KERNELS)
    endif ()

endmacro()
