# Source
set(SRC 
	face_seg.cpp
	utilities.cpp
)
set(HDR 
	face_seg/face_seg.h
	face_seg/utilities.h
)

# Target
add_library(face_seg ${SRC} ${HDR})
target_include_directories(face_seg PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	${OpenCV_INCLUDE_DIRS}
	${Caffe_INCLUDE_DIRS}
)
target_link_libraries(face_seg PUBLIC
	${OpenCV_LIBS}
	${Caffe_LIBRARIES}
)

# Installations
install(TARGETS face_seg
	 EXPORT face_seg-targets
     RUNTIME DESTINATION bin COMPONENT dev
     LIBRARY DESTINATION lib COMPONENT dev
     ARCHIVE DESTINATION lib COMPONENT dev)
install(FILES ${HDR} DESTINATION include/face_seg)
