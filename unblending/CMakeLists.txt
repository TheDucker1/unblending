find_package(Eigen3 REQUIRED)
if((NOT TARGET Eigen3::Eigen) AND (DEFINED EIGEN3_INCLUDE_DIR))
	# Eigen 3.0, 3.1, 3.2 do not provide the target named Eigen3::Eigen
	add_library(AliasEigen3 INTERFACE)
	target_include_directories(AliasEigen3 INTERFACE ${EIGEN3_INCLUDE_DIR})
	add_library(Eigen3::Eigen ALIAS AliasEigen3)
endif()

find_package(Threads REQUIRED)

file(GLOB headers include/unblending/*.hpp)
file(GLOB sources src/*.cpp)
add_library(unblending STATIC ${headers} ${sources})
target_link_libraries(unblending Eigen3::Eigen nlopt Threads::Threads json11 tinycolormap timer parallel-util nlopt-util)
target_include_directories(
	unblending
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

if(UNBLENDING_ENABLE_CUDA)
	# Ship the fixed FP32 blend header used by the solver.
	target_sources(unblending PRIVATE src/cuda/pixel_pipeline.cu src/cuda/compositing/blend_fp32.hpp)
	target_include_directories(unblending PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")
	target_compile_definitions(unblending PRIVATE UNBLENDING_HAS_CUDA)
	set_target_properties(unblending PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
	target_compile_options(unblending PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
	if(MSVC)
		target_compile_options(unblending PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:strict>)
	else()
		target_compile_options(unblending PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
	endif()
endif()

install(FILES ${headers} DESTINATION include/unblending)
install(TARGETS unblending ARCHIVE DESTINATION lib)

if(UNBLENDING_BUILD_SDK)
	add_library(unblending_sdk SHARED sdk/unblending_c.cpp)
	target_link_libraries(unblending_sdk PRIVATE unblending)
	target_include_directories(unblending_sdk PRIVATE src)
	target_compile_definitions(unblending_sdk PRIVATE UNBLENDING_SDK_EXPORTS)
	set_target_properties(unblending_sdk PROPERTIES
		OUTPUT_NAME unblending
		ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sdk-lib"
		VERSION 1.0.0 SOVERSION 1
		CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
	)
	if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
		set(sdk_exports "${CMAKE_CURRENT_SOURCE_DIR}/sdk/exports.map")
		target_link_options(unblending_sdk PRIVATE "LINKER:--version-script,${sdk_exports}")
		set_property(TARGET unblending_sdk APPEND PROPERTY LINK_DEPENDS "${sdk_exports}")
	endif()
	install(TARGETS unblending_sdk
		RUNTIME DESTINATION bin COMPONENT unblending_sdk
		LIBRARY DESTINATION lib COMPONENT unblending_sdk
		ARCHIVE DESTINATION lib COMPONENT unblending_sdk
	)
	install(FILES include/unblending.h DESTINATION include COMPONENT unblending_sdk)
	install(FILES "${PROJECT_SOURCE_DIR}/LICENSE.LGPLv3" "${PROJECT_SOURCE_DIR}/LICENSE.Commercial"
		DESTINATION share/licenses/unblending COMPONENT unblending_sdk)
endif()

if(UNBLENDING_BUILD_IMAGE_IO OR UNBLENDING_BUILD_CLI_APP OR UNBLENDING_BUILD_GUI_APP)
	find_package(Qt5Gui REQUIRED)
	add_library(unblending_image_io STATIC io/image_qt.cpp)
	target_link_libraries(unblending_image_io PUBLIC unblending PRIVATE Qt5::Gui)
	install(TARGETS unblending_image_io ARCHIVE DESTINATION lib)
endif()
