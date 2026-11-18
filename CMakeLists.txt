cmake_minimum_required(VERSION 3.20)
project(vs_ngu VERSION 0.1.1 LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(VS IMPORTED_TARGET vapoursynth>=82)
endif()

if(NOT TARGET PkgConfig::VS)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import os, vapoursynth; print(os.path.join(os.path.dirname(vapoursynth.__file__), 'include'))"
        OUTPUT_VARIABLE VAPOURSYNTH_PY_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    find_path(VAPOURSYNTH_INCLUDE_DIR
        NAMES VapourSynth4.h
        HINTS "${VAPOURSYNTH_PY_INCLUDE}"
        PATH_SUFFIXES vapoursynth/include vapoursynth include
    )
    if(NOT VAPOURSYNTH_INCLUDE_DIR)
        message(FATAL_ERROR "VapourSynth4.h not found. Ensure vapoursynth>=82 is installed.")
    endif()
    add_library(PkgConfig::VS INTERFACE IMPORTED)
    target_include_directories(PkgConfig::VS INTERFACE "${VAPOURSYNTH_INCLUDE_DIR}")
endif()

if(WIN32)
    # Windows native crypto via BCrypt (CNG)
else()
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
endif()

if(NOT SKBUILD)
    include(CTest)
endif()
include(GNUInstallDirs)

option(NGU_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(NGU_NATIVE_ARCH "Optimize for the build machine's CPU (not portable)" ON)
option(NGU_DUMP_SHADERS "Enable opt-in NGU_SHADER_DUMP_DIR precision audit dumps" OFF)
add_library(ngu_build_options INTERFACE)
target_compile_features(ngu_build_options INTERFACE cxx_std_20)
if(MSVC)
    target_compile_options(ngu_build_options INTERFACE /W4 /fp:strict /utf-8)
    target_compile_definitions(ngu_build_options INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN)
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
if(WIN32)
    target_link_libraries(ngu_cpu PUBLIC ngu_build_options PRIVATE bcrypt)
else()
    target_link_libraries(ngu_cpu PUBLIC ngu_build_options PRIVATE OpenSSL::Crypto)
endif()

find_path(VULKAN_INCLUDE_DIR NAMES vulkan/vulkan_core.h)
set(VULKAN_1_4_FOUND FALSE)
if(VULKAN_INCLUDE_DIR)
    file(READ "${VULKAN_INCLUDE_DIR}/vulkan/vulkan_core.h" VULKAN_CORE_H)
    if(VULKAN_CORE_H MATCHES "VK_VERSION_1_4")
        set(VULKAN_1_4_FOUND TRUE)
    endif()
endif()

if(NOT VULKAN_1_4_FOUND)
    message(STATUS "Vulkan 1.4 headers not found, fetching Vulkan-Headers v1.4.304 via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        vulkan_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG v1.4.304
    )
    # Use the headers for compilation only; do not install dependency files.
    set(VULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(vulkan_headers)
endif()

add_library(ngu_vulkan STATIC src/ngu_vulkan.cpp)
set_target_properties(ngu_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_link_libraries(ngu_vulkan PUBLIC ngu_cpu PkgConfig::VS)
if(TARGET Vulkan-Headers)
    target_link_libraries(ngu_vulkan PUBLIC Vulkan-Headers)
elseif(TARGET Vulkan::Headers)
    target_link_libraries(ngu_vulkan PUBLIC Vulkan::Headers)
elseif(VULKAN_INCLUDE_DIR)
    target_include_directories(ngu_vulkan PUBLIC "${VULKAN_INCLUDE_DIR}")
endif()
target_compile_definitions(ngu_vulkan PUBLIC VS_USE_API_43)
if(NGU_DUMP_SHADERS)
    target_compile_definitions(ngu_vulkan PRIVATE NGU_DUMP_SHADERS)
endif()

add_library(vsngu MODULE src/vs_plugin.cpp)
set_target_properties(vsngu PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_link_libraries(vsngu PRIVATE ngu_vulkan)
target_compile_definitions(vsngu PRIVATE VS_USE_API_43)

if(SKBUILD)
    install(TARGETS vsngu
        LIBRARY DESTINATION "vapoursynth/plugins/vsngu"
        RUNTIME DESTINATION "vapoursynth/plugins/vsngu"
    )
else()
    install(TARGETS vsngu
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/vapoursynth/vsngu"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/vapoursynth/vsngu"
    )
endif()
# No weights, original bytecode, PE files or research products are installed.

if(BUILD_TESTING AND NOT SKBUILD)
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
