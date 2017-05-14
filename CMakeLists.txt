cmake_minimum_required(VERSION 3.4)
project(face_segmentation)

# Set version
set(FACE_SEG_MAJOR_VERSION 0)
set(FACE_SEG_MINOR_VERSION 9)
set(FACE_SEG_PATCH_VERSION 0)
set(FACE_SEG_VERSION ${FACE_SEG_MAJOR_VERSION}.${FACE_SEG_MINOR_VERSION}.${FACE_SEG_PATCH_VERSION})

# Global configurations
# ===================================================
if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
	set(DEF_INSTALL_CMAKE_DIR cmake)
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC")
	set(DEF_INSTALL_CMAKE_DIR lib/cmake/face_seg)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
	"Installation directory for CMake files")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_DEBUG_POSTFIX "_d")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Optional 3rd party components
# ===================================================
option(WITH_BOOST_STATIC "Boost static libraries" ON)

# Build components
# ===================================================
option(BUILD_DOCS "Build documentation using Doxygen" ON)
option(BUILD_INTERFACE_PYTHON "Build interface for Python" ON)

# Find dependencies
# ===================================================

# Boost
set(Boost_USE_STATIC_LIBS ${WITH_BOOST_STATIC})
set(BOOST_ALL_DYN_LINK NOT ${WITH_BOOST_STATIC})
if(WIN32)
	if(${WITH_BOOST_STATIC})
		add_definitions(-DBOOST_ALL_NO_LIB)
	else()
		add_definitions(-DBOOST_ALL_DYN_LINK)
	endif()
endif()
find_package(Boost COMPONENTS filesystem program_options regex timer)

# OpenCV
find_package(OpenCV REQUIRED highgui imgproc imgcodecs calib3d photo)

# Caffe
find_package(Caffe REQUIRED)

# Doxygen
find_package(Doxygen)

# Interfaces

# Add sub-directories
# ===================================================
add_subdirectory(face_seg)
add_subdirectory(face_seg_image)
add_subdirectory(face_seg_batch)

# Interfaces
if(BUILD_INTERFACE_PYTHON)
	add_subdirectory(interfaces/python)
endif()

# Documentation
if(BUILD_DOCS)
	add_subdirectory(doc)
endif()

# Export configuration
# ===================================================

# Add all targets to the build-tree export set
set(FACE_SEG_TARGETS face_seg face_seg_image face_seg_batch)
export(TARGETS ${FACE_SEG_TARGETS}
  FILE "${PROJECT_BINARY_DIR}/face_seg-targets.cmake")
  
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE face_seg)
	 
# Create config files
configure_file(cmake/face_segmentation-config.cmake.in
  "${PROJECT_BINARY_DIR}/face_segmentation-config.cmake" @ONLY)
configure_file(cmake/face_segmentation-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/face_segmentation-config-version.cmake" @ONLY)
  
# Install config files
install(FILES
  "${PROJECT_BINARY_DIR}/face_segmentation-config.cmake"
  "${PROJECT_BINARY_DIR}/face_segmentation-config-version.cmake"
  DESTINATION "cmake" COMPONENT dev)
  
# Install the export set for use with the install-tree
install(EXPORT face_seg-targets DESTINATION cmake COMPONENT dev)

# Install sample data
file(GLOB SAMPLE_IMAGES "data/images/*.jpg")
install(FILES ${SAMPLE_IMAGES} DESTINATION data/images)