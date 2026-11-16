cmake_minimum_required(VERSION 3.20)
project(vs_ngu VERSION 0.1.0 LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(PkgConfig REQUIRED)
pkg_check_modules(VS REQUIRED IMPORTED_TARGET vapoursynth>=82)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
include(CTest)
include(GNUInstallDirs)

option(NGU_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(NGU_NATIVE_ARCH "Optimize for the build machine's CPU (not portable)" ON)
option(NGU_DUMP_SHADERS "Enable opt-in NGU_SHADER_DUMP_DIR precision audit dumps" OFF)
add_library(ngu_build_options INTERFACE)
target_compile_features(ngu_build_options INTERFACE cxx_std_20)
if(MSVC)
    target_compile_options(ngu_build_options INTERFACE /W4 /WX /fp:strict)
else()
    target_compile_options(ngu_build_options INTERFACE
        -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off)
    if(NGU_NATIVE_ARCH)
        target_compile_options(ngu_build_options INTERFACE -march=native)
    endif()
    if(NGU_SANITIZE)
        target_compile_options(ngu_build_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(ngu_build_options INTERFACE -fsanitize=address,undefined)
    endif()
endif()

set(catalog_header "${CMAKE_CURRENT_BINARY_DIR}/generated/ngu_catalog.hpp")
add_custom_command(OUTPUT "${catalog_header}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ngu_catalog.py"
        --index "${CMAKE_CURRENT_SOURCE_DIR}/models/madvr09217-index.json"
        --fp16-index "${CMAKE_CURRENT_SOURCE_DIR}/models/madvr09217-fp16-index.json"
        --output "${catalog_header}"
    DEPENDS tools/generate_ngu_catalog.py models/madvr09217-index.json models/madvr09217-fp16-index.json
    VERBATIM)

add_library(ngu_cpu STATIC src/ngu_cpu.cpp src/ngu_model.cpp "${catalog_header}")
set_target_properties(ngu_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_include_directories(ngu_cpu PUBLIC src PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(ngu_cpu PUBLIC ngu_build_options PRIVATE OpenSSL::Crypto)

add_library(ngu_vulkan STATIC src/ngu_vulkan.cpp)
set_target_properties(ngu_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_link_libraries(ngu_vulkan PUBLIC ngu_cpu PkgConfig::VS)
target_compile_definitions(ngu_vulkan PUBLIC VS_USE_API_43)
if(NGU_DUMP_SHADERS)
    target_compile_definitions(ngu_vulkan PRIVATE NGU_DUMP_SHADERS)
endif()

add_library(vsngu MODULE src/vs_plugin.cpp)
set_target_properties(vsngu PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_link_libraries(vsngu PRIVATE ngu_vulkan)
target_compile_definitions(vsngu PRIVATE VS_USE_API_43)
install(TARGETS vsngu LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/vapoursynth")
# No weights, original bytecode, PE files or research products are installed.

if(BUILD_TESTING)
    set(NGU_WEIGHTS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/weights" CACHE PATH
        "Common root containing weights/ (FP32) and weights-fp16/ (FP16)")
    set(NGU_MADVR_SOURCE "" CACHE FILEPATH
        "Optional authorized madVR64.ax 0.92.17 for extraction/conversion round-trip tests")
    add_executable(ngu_native_tests tests/native_cpu.cpp)
    target_link_libraries(ngu_native_tests PRIVATE ngu_cpu)
    add_test(NAME native_math COMMAND ngu_native_tests)
    add_test(NAME half_fma_reference COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/half_reference.py" "$<TARGET_FILE:ngu_native_tests>")
    add_test(NAME native_archive COMMAND ngu_native_tests "${NGU_WEIGHTS_ROOT}")
    add_test(NAME fp16_archive COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fp16_archive.py"
        --source "${NGU_WEIGHTS_ROOT}/weights"
        --archive "${NGU_WEIGHTS_ROOT}/weights-fp16")
    set(layout_test_args --source "${NGU_WEIGHTS_ROOT}/weights"
        --cmake "${CMAKE_COMMAND}" --generator "${CMAKE_GENERATOR}")
    if(NGU_MADVR_SOURCE)
        list(APPEND layout_test_args --madvr "${NGU_MADVR_SOURCE}")
    endif()
    add_test(NAME weights_layout COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/weights_layout.py" ${layout_test_args})

    find_library(VS_CORE_LIBRARY NAMES vapoursynth REQUIRED)
    # Python-package installs may expose /usr/lib symlinks to the actual core.
    # The core locates its standard-filter library next to its loaded path.
    # Link the real location so CMake supplies that directory in BUILD_RPATH.
    get_filename_component(VS_CORE_REALPATH "${VS_CORE_LIBRARY}" REALPATH)
    add_executable(ngu_vs_tests tests/native_vs.cpp)
    target_compile_definitions(ngu_vs_tests PRIVATE VS_USE_API_43)
    target_link_libraries(ngu_vs_tests PRIVATE ngu_vulkan "${VS_CORE_REALPATH}")

    add_executable(ngu_precision_bench tools/benchmark_precision.cpp)
    target_compile_definitions(ngu_precision_bench PRIVATE VS_USE_API_43)
    target_link_libraries(ngu_precision_bench PRIVATE ngu_build_options PkgConfig::VS "${VS_CORE_REALPATH}")
    add_test(NAME native_vs_lifecycle COMMAND ngu_vs_tests "$<TARGET_FILE:vsngu>" "${NGU_WEIGHTS_ROOT}")
    add_test(NAME native_vs_large COMMAND ngu_vs_tests "$<TARGET_FILE:vsngu>" "${NGU_WEIGHTS_ROOT}" --large)
    set_tests_properties(native_vs_lifecycle native_vs_large PROPERTIES TIMEOUT 300)
    if(NGU_SANITIZE)
        # The Debug scalar half oracle plus ASan now approaches five minutes.
        set_tests_properties(native_vs_lifecycle native_vs_large PROPERTIES TIMEOUT 600)
    endif()

    # The Python host is not sanitizer-linked. Sanitizer builds exercise FP16
    # via the native math/archive/lifecycle tests, including GPU retained frames.
    if(NOT NGU_SANITIZE)
        add_test(NAME fp16_plugin COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/fp16_plugin.py"
            --plugin "$<TARGET_FILE:vsngu>" --weights "${NGU_WEIGHTS_ROOT}")
        set_tests_properties(fp16_plugin PROPERTIES TIMEOUT 300)
    endif()
endif()
