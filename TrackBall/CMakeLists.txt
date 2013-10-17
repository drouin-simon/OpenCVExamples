cmake_minimum_required(VERSION 2.8)
project( TrackBall )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( TrackBall main.cpp )
target_link_libraries( TrackBall ${OpenCV_LIBS} )
