cmake_minimum_required(VERSION 3.16)
project(LULESH LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_SOURCE_DIR}/config.cmake)

option(Charm_ENABLE_GPU "Build LULESH with Charm++/Kokkos GPU backend" ON)

set(Kokkos_ROOT ${KOKKOS_DIR})
find_package(Kokkos 4.2 REQUIRED CONFIG)
#find_package(MPI REQUIRED COMPONENTS CXX)

# --- START OF FIX ---
# 1. Strip the flag from variables set in config.cmake
string(REPLACE "-Wext-lambda-captures-this" "" CPU_OPTS "${CPU_OPTS}")

# 2. Strip the flag from global CMake flags
string(REPLACE "-Wext-lambda-captures-this" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# 3. Strip the flag from the imported Kokkos target itself
if(TARGET Kokkos::kokkos)
    get_target_property(KOKKOS_OPTIONS Kokkos::kokkos INTERFACE_COMPILE_OPTIONS)
    if(KOKKOS_OPTIONS)
        list(REMOVE_ITEM KOKKOS_OPTIONS "-Wext-lambda-captures-this")
        set_target_properties(Kokkos::kokkos PROPERTIES INTERFACE_COMPILE_OPTIONS "${KOKKOS_OPTIONS}")
    endif()
endif()
# --- END OF FIX ---

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LULESH_SOURCES
    lulesh-comm.cc
    lulesh-init.cc
    lulesh-util.cc
    lulesh-viz.cc
    lulesh.cc
)

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)

add_custom_command(
   OUTPUT ${GENERATED_DIR}/lulesh.decl.h ${GENERATED_DIR}/lulesh.def.h
   COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/lulesh.ci ${GENERATED_DIR}/lulesh.ci
   COMMAND ${CMAKE_COMMAND} -E chdir ${GENERATED_DIR} ${CHARMC} lulesh.ci
   DEPENDS ${CMAKE_SOURCE_DIR}/lulesh.ci
   COMMENT "Generating Charm++ interface for lulesh"
   VERBATIM
)

if(Charm_ENABLE_GPU)
    set(CMAKE_CXX_COMPILER "${CHARMC}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -noswitcherror")
    add_executable(lulesh ${LULESH_SOURCES} ${GENERATED_DIR}/lulesh.decl.h ${GENERATED_DIR}/lulesh.def.h)
    target_include_directories(lulesh PRIVATE ${CMAKE_SOURCE_DIR} ${GENERATED_DIR} ${CHARM_DIR}/include)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPU_OPTS} ${LD_OPTS}")
    target_link_libraries(lulesh PRIVATE Kokkos::kokkos)
endif()
