# External cuDNN only: do not download or redistribute its binaries.
include(FindPackageHandleStandardArgs)
set(CUDNN_ROOT "" CACHE PATH "Root of the cuDNN installation")
find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h
    HINTS "${CUDNN_ROOT}" ENV CUDNN_ROOT PATH_SUFFIXES include)
# NVIDIA wheels ship the ABI-versioned soname without an unversioned symlink.
find_library(CUDNN_LIBRARY NAMES cudnn cudnn64_9 libcudnn.so.9
    HINTS "${CUDNN_ROOT}" ENV CUDNN_ROOT PATH_SUFFIXES lib lib64 lib/x64)
unset(CUDNN_VERSION)
if(CUDNN_INCLUDE_DIR AND EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(STRINGS "${CUDNN_INCLUDE_DIR}/cudnn_version.h" _cudnn_version_lines
        REGEX "^#define[ \t]+CUDNN_(MAJOR|MINOR|PATCHLEVEL)[ \t]+")
    set(_cudnn_version_parts)
    foreach(_part MAJOR MINOR PATCHLEVEL)
        string(REGEX MATCH "#define[ \t]+CUDNN_${_part}[ \t]+([0-9]+)" _match "${_cudnn_version_lines}")
        if(NOT _match)
            set(_cudnn_version_parts)
            break()
        endif()
        list(APPEND _cudnn_version_parts "${CMAKE_MATCH_1}")
    endforeach()
    list(LENGTH _cudnn_version_parts _cudnn_version_count)
    if(_cudnn_version_count EQUAL 3)
        list(JOIN _cudnn_version_parts "." CUDNN_VERSION)
    endif()
endif()
find_package_handle_standard_args(CUDNN
    REQUIRED_VARS CUDNN_INCLUDE_DIR CUDNN_LIBRARY CUDNN_VERSION
    VERSION_VAR CUDNN_VERSION)
if(CUDNN_FOUND AND NOT TARGET CUDNN::cudnn)
    add_library(CUDNN::cudnn UNKNOWN IMPORTED)
    set_target_properties(CUDNN::cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
endif()
mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
