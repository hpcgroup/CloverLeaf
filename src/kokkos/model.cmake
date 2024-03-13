macro(setup)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_EXTENSIONS OFF)

    find_package(Kokkos REQUIRED)

    register_link_library(Kokkos::kokkos)

    if (${KOKKOS_BACK_END} STREQUAL "CUDA")
        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -extended-lambda -Wext-lambda-captures-this -expt-relaxed-constexpr")

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CUDA)
    elseif (${KOKKOS_BACK_END} STREQUAL "HIP")
        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)

        set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
    endif ()

endmacro()
