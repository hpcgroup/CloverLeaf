
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection, this is used for host compilation when required by the SYCL compiler"
        "c++")

register_flag_required(SYCL_COMPILER
        "Compile using the specified SYCL compiler implementation
        Supported values are
           ONEAPI-ICPX  - icpx as a standalone compiler
           ONEAPI-Clang - oneAPI's Clang driver (enabled via `source /opt/intel/oneapi/setvars.sh  --include-intel-llvm`)
           DPCPP        - dpc++ as a standalone compiler (https://github.com/intel/llvm)
           HIPSYCL      - hipSYCL compiler (https://github.com/illuhad/hipSYCL)
           COMPUTECPP   - ComputeCpp compiler (https://developer.codeplay.com/products/computecpp/ce/home)")

register_flag_optional(SYCL_COMPILER_DIR
        "Absolute path to the selected SYCL compiler directory, most are packaged differently so set the path according to `SYCL_COMPILER`:
           ONEAPI-ICPX              - `icpx` must be used for OneAPI 2023 and later on releases (i.e `source /opt/intel/oneapi/setvars.sh` first)
           ONEAPI-Clang             - set to the directory that contains the Intel clang++ binary.
           HIPSYCL|DPCPP|COMPUTECPP - set to the root of the binary distribution that contains at least `bin/`, `include/`, and `lib/`"
        "")

register_flag_optional(USE_RANGE2D_MODE
        "Set range<2> indexing mode, supported values are:
           RANGE2D_NORMAL - Leave range<2> as-is
           RANGE2D_LINEAR - Linearise range<2> to a range<1> using divisions and modulo
           RANGE2D_ROUND  - Round all dimensions in a range<2> to multiples of 32"
        "RANGE2D_LINEAR")

register_flag_optional(USE_SYCL2020_REDUCTION
        "Whether to use reduction introduced in SYCO2020"
        "ON")


register_flag_optional(USE_HOSTTASK
        "Whether to use SYCL2020 host_task for MPI related calls or fallback to queue.wait() not all SYCL compilers support this"
        "OFF")

register_flag_optional(OpenCL_LIBRARY
        "[ComputeCpp only] Path to OpenCL library, usually called libOpenCL.so"
        "${OpenCL_LIBRARY}")

macro(setup)
    set(CMAKE_CXX_STANDARD 17)

    if (USE_RANGE2D_MODE)
        register_definitions(RANGE2D_MODE=${USE_RANGE2D_MODE})
    endif ()

    if (USE_SYCL2020_REDUCTION)
        register_definitions(USE_SYCL2020_REDUCTION)
    endif ()

    if (USE_HOSTTASK)
        register_definitions(USE_HOSTTASK)
    endif ()

    if (${SYCL_COMPILER} STREQUAL "HIPSYCL")

        set(hipSYCL_DIR ${SYCL_COMPILER_DIR}/lib/cmake/hipSYCL)

        if (NOT EXISTS "${hipSYCL_DIR}")
            message(WARNING "Falling back to hipSYCL < 0.9.0 CMake structure")
            set(hipSYCL_DIR ${SYCL_COMPILER_DIR}/lib/cmake)
        endif ()
        if (NOT EXISTS "${hipSYCL_DIR}")
            message(FATAL_ERROR "Can't find the appropriate CMake definitions for hipSYCL")
        endif ()

        # register_definitions(_GLIBCXX_USE_CXX11_ABI=0)
        find_package(hipSYCL CONFIG REQUIRED)
        message(STATUS "ok")

    elseif (${SYCL_COMPILER} STREQUAL "COMPUTECPP")

        list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
        set(ComputeCpp_DIR ${SYCL_COMPILER_DIR})

        # don't point to the CL dir as the imports already have the CL prefix
        set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}")

        register_definitions(CL_TARGET_OPENCL_VERSION=220 _GLIBCXX_USE_CXX11_ABI=0)
        # ComputeCpp needs OpenCL
        find_package(ComputeCpp REQUIRED)

        # this must come after FindComputeCpp (!)
        set(COMPUTECPP_USER_FLAGS -O3 -no-serial-memop)

    elseif (${SYCL_COMPILER} STREQUAL "DPCPP")
        register_append_cxx_flags(ANY -fsycl)
        register_append_link_flags(-fsycl)
    elseif (${SYCL_COMPILER} STREQUAL "ONEAPI-ICPX")
        register_append_cxx_flags(ANY -fsycl)
        register_append_link_flags(-fsycl)
    elseif (${SYCL_COMPILER} STREQUAL "ONEAPI-Clang")
        register_append_cxx_flags(ANY -fsycl)
        register_append_link_flags(-fsycl)
    else ()
        message(FATAL_ERROR "SYCL_COMPILER=${SYCL_COMPILER} is unsupported")
    endif ()

    list(APPEND IMPL_SOURCES
            src/${MODEL_LOWER}/update_tile_halo_kernel_t.cpp
            src/${MODEL_LOWER}/update_tile_halo_kernel_l.cpp
            src/${MODEL_LOWER}/update_tile_halo_kernel_r.cpp
            src/${MODEL_LOWER}/update_tile_halo_kernel_b.cpp
            src/${MODEL_LOWER}/update_halo_1.cpp
            src/${MODEL_LOWER}/update_halo_2.cpp
    )

endmacro()


macro(setup_target NAME)
    if (
    (${SYCL_COMPILER} STREQUAL "COMPUTECPP") OR
    (${SYCL_COMPILER} STREQUAL "HIPSYCL"))
        # so ComputeCpp and hipSYCL has this weird (and bad) CMake usage where they append their
        # own custom integration header flags AFTER the target has been specified
        # hence this macro here
        add_sycl_to_target(
                TARGET ${NAME}
                SOURCES ${IMPL_SOURCES})
    endif ()
endmacro()
