# Validation
if(NOT Boost_FOUND)
	message(STATUS "face_seg_batch won't be built because Boost is missing.")
	return()
endif()

if(find_face_landmarks_FOUND AND dlib_FOUND)
	add_definitions(-DWITH_FIND_FACE_LANDMARKS)
endif()

# Target
add_executable(face_seg_batch face_seg_batch.cpp)
target_include_directories(face_seg_batch PRIVATE 
	${Boost_INCLUDE_DIRS}
)
target_link_libraries(face_seg_batch PRIVATE
	face_seg
	${Boost_LIBRARIES}
)

if(find_face_landmarks_FOUND AND dlib_FOUND)
	target_include_directories(face_seg_batch PRIVATE 
		${FIND_FACE_LANDMARKS_INCLUDE_DIRS}
	)
	target_link_libraries(face_seg_batch PRIVATE
		${FIND_FACE_LANDMARKS_LIBRARIES}
	)
endif()

# Installations
install(TARGETS face_seg_batch EXPORT face_seg-targets DESTINATION bin COMPONENT app)
install(FILES face_seg_batch.cfg DESTINATION bin COMPONENT app)
#set(FACE_SEG_TARGETS ${FACE_SEG_TARGETS} face_seg_batch)
