# additional config
set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
set(DOXYGEN_INPUT_PATH "\"${CMAKE_SOURCE_DIR}/face_seg/face_seg\"")
set(DOXYGEN_OUTPUT_PATH "\"${CMAKE_CURRENT_BINARY_DIR}/doxygen\"")
set(DOXYGEN_PROJECT_NAME "\"Nirkin Face Segmentation\"")
set(DOXYGEN_PROJECT_BRIEF "\"Deep face segmentation based on fully connected convolutional neural network.\"")

# Write doxygen configuration file
configure_file(Doxyfile.in ${doxyfile} @ONLY)

add_custom_target(doxygen
	COMMAND ${CMAKE_COMMAND} -E echo "Building API Documentation..."
	COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
	COMMAND ${CMAKE_COMMAND} -E echo "Done."
	DEPENDS ${doxyfile}
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
	DESTINATION doc
	COMPONENT "docs" OPTIONAL
)