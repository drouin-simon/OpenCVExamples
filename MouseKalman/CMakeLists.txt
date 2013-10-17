cmake_minimum_required(VERSION 2.8)
project( MouseKalman )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( MouseKalman main.cpp )
target_link_libraries( MouseKalman ${OpenCV_LIBS} )
