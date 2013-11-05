cmake_minimum_required(VERSION 2.8)
project( TraXing )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( TraXing main.cpp crossingtracker.h crossingtracker.cpp )
target_link_libraries( TraXing ${OpenCV_LIBS} )
