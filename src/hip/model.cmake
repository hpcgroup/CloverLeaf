
register_flag_required(CMAKE_CXX_COMPILER
        "Absolute path to the AMD HIP C++ compiler")

register_flag_optional(MANAGED_ALLOC "Use UVM (hipMallocManaged) instead of the device-only allocation (hipMalloc)"
        "OFF")

register_flag_optional(SYNC_ALL_KERNELS
        "Fully synchronise all kernels after launch, this also enables synchronous error checking with line and file name"
        "OFF")


macro(setup)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_HIP_STANDARD 17)

    enable_language(HIP)
    set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)

    if (MANAGED_ALLOC)
        register_definitions(CLOVER_MANAGED_ALLOC)
    endif ()

    if (SYNC_ALL_KERNELS)
        register_definitions(CLOVER_SYNC_ALL_KERNELS)
    endif ()


endmacro()
