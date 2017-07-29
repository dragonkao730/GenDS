FIND_PATH(GenDS_INCLUDE_DIR gends.h
    /usr/include/gends
    /usr/local/include/gends)

FIND_LIBRARY(GenDS_LIBRARY
    NAMES gends
    PATH /usr/lib /usr/local/lib)

IF (GenDS_INCLUDE_DIR AND GenDS_LIBRARY)
    SET(GenDS_FOUND TRUE)
ENDIF (GenDS_INCLUDE_DIR AND GenDS_LIBRARY)

IF (GenDS_FOUND)
    IF (NOT GenDS_FIND_QUIETLY)
        MESSAGE(STATUS "Found GenDS: ${GenDS_LIBRARY}")
    ENDIF (NOT GenDS_FIND_QUIETLY)
ELSE (GenDS_FOUND)
    IF (GenDS_FIND_QUIETLY)
        MESSAGE(FATAL_ERROR "Could not find GenDS library")
    ENDIF (GenDS_FIND_QUIETLY)
ENDIF (GenDS_FOUND)