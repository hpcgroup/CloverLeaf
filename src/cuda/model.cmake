register_flag_optional(CUDA_EXTRA_FLAGS
        "Additional CUDA flags passed to nvcc, this is appended after `CUDA_ARCH`"
        "")

register_flag_optional(MANAGED_ALLOC "Use UVM (cudaMallocManaged) instead of the device-only allocation (cudaMalloc)"
        "OFF")

register_flag_optional(SYNC_ALL_KERNELS
        "Fully synchronise all kernels after launch, this also enables synchronous error checking with line and file name"
        "OFF")

macro(setup)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)

    enable_language(CUDA)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -extended-lambda -use_fast_math -restrict -keep ${CUDA_EXTRA_FLAGS}")

    # CMake defaults to -O2 for CUDA at Release, let's wipe that and use the global RELEASE_FLAG
    # appended later
    wipe_gcc_style_optimisation_flags(CMAKE_CUDA_FLAGS_${BUILD_TYPE})

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
    foreach (SRC ${PROJECT_SRC})
        set_source_files_properties("${SRC}" PROPERTIES LANGUAGE CUDA)
    endforeach ()
endmacro()
