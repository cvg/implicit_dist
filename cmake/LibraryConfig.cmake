# Target
add_library(${LIBRARY_NAME}
  ${SOURCES}
  ${HEADERS_PUBLIC}
  ${HEADERS_PRIVATE}
  )

# Alias:
add_library(${PROJECT_NAME}::${LIBRARY_NAME} ALIAS ${LIBRARY_NAME})

# C++17
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)

# Add definitions for targets
target_compile_definitions(${LIBRARY_NAME} PUBLIC
  "${PROJECT_NAME_UPPERCASE}_DEBUG=$<CONFIG:Debug>")

# Global includes. Used by all targets
target_include_directories(
  ${LIBRARY_NAME} PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${GENERATED_HEADERS_DIR}>"
    "$<INSTALL_INTERFACE:.>"
)

# Targets:
#   - <prefix>/lib/libimplicit_dist.a
#   - header location after install: <prefix>/implicit_dist/implicit_dist.h
#   - headers can be included by C++ code `#include <implicit_dist/implicit_dist.h>`
install(
    TARGETS              "${LIBRARY_NAME}"
    EXPORT               "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION  "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION  "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION  "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)


# Headers:
#   - implicit_dist/*.h -> <prefix>/include/implicit_dist/*.h
foreach ( file ${HEADERS_PUBLIC} )
    get_filename_component( dir ${file} DIRECTORY )
    install( FILES ${file} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_FOLDER}/${dir}" )
endforeach()
#install( FILES "${GENERATED_HEADERS_DIR}/implicit_dist/implicit_dist.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_FOLDER}")

# Headers:
#   - generated_headers/implicit_dist/version.h -> <prefix>/include/implicit_dist/version.h
install(
    FILES       "${GENERATED_HEADERS_DIR}/${LIBRARY_FOLDER}/version.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_FOLDER}"
)

# Config
#   - <prefix>/lib/cmake/implicit_dist/implicit_distConfig.cmake
#   - <prefix>/lib/cmake/implicit_dist/implicit_distConfigVersion.cmake
install(
    FILES       "${PROJECT_CONFIG_FILE}"
                "${VERSION_CONFIG_FILE}"
    DESTINATION "${CONFIG_INSTALL_DIR}"
)

# Config
#   - <prefix>/lib/cmake/implicit_dist/implicit_distTargets.cmake
install(
  EXPORT      "${TARGETS_EXPORT_NAME}"
  FILE        "${PROJECT_NAME}Targets.cmake"
  DESTINATION "${CONFIG_INSTALL_DIR}"
  NAMESPACE   "${PROJECT_NAME}::"
)
