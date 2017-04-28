cmake_minimum_required(VERSION 3.6)
project(engine)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES engine.cc easy_image.cc l_parser.cc ini_configuration.cc vector.cc)
add_executable(engine ${SOURCE_FILES})