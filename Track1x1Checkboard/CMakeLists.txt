cmake_minimum_required(VERSION 2.8)
project( Track1x1Checkboard )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( Track1x1Checkboard main.cpp )
target_link_libraries( Track1x1Checkboard ${OpenCV_LIBS} )
