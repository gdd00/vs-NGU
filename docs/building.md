# Building

Requirements:

- CMake 3.20+ and a C++20 compiler
- VapourSynth R82+ headers, found through pkg-config or the `vapoursynth` Python package
- OpenSSL (libcrypto). Linux only; Windows uses BCrypt.
- Vulkan 1.4 headers. If they are missing, CMake fetches them automatically.

```bash
cmake -S . -B build/native -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D NGU_NATIVE_ARCH=ON \
    -D BUILD_TESTING=OFF
cmake --build build/native
cmake --install build/native
```

The build output is `build/native/vsngu.so` on Linux or `build/native/vsngu.dll` on Windows. Native installation uses `${CMAKE_INSTALL_LIBDIR}/vapoursynth/vsngu/` for library artifacts and `${CMAKE_INSTALL_BINDIR}/vapoursynth/vsngu/` for runtime artifacts. The default weights root is still the `weights/` directory next to the installed library.

To build a wheel locally:

```bash
pip install .
```
