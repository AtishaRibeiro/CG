# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/atisha/CLion/clion-2017.1.1/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/atisha/CLion/clion-2017.1.1/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/atisha/Computer Graphics/engine_old"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/atisha/Computer Graphics/engine_old/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/engine.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/engine.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/engine.dir/flags.make

CMakeFiles/engine.dir/engine.cc.o: CMakeFiles/engine.dir/flags.make
CMakeFiles/engine.dir/engine.cc.o: ../engine.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/atisha/Computer Graphics/engine_old/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/engine.dir/engine.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/engine.dir/engine.cc.o -c "/home/atisha/Computer Graphics/engine_old/engine.cc"

CMakeFiles/engine.dir/engine.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/engine.dir/engine.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/atisha/Computer Graphics/engine_old/engine.cc" > CMakeFiles/engine.dir/engine.cc.i

CMakeFiles/engine.dir/engine.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/engine.dir/engine.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/atisha/Computer Graphics/engine_old/engine.cc" -o CMakeFiles/engine.dir/engine.cc.s

CMakeFiles/engine.dir/engine.cc.o.requires:

.PHONY : CMakeFiles/engine.dir/engine.cc.o.requires

CMakeFiles/engine.dir/engine.cc.o.provides: CMakeFiles/engine.dir/engine.cc.o.requires
	$(MAKE) -f CMakeFiles/engine.dir/build.make CMakeFiles/engine.dir/engine.cc.o.provides.build
.PHONY : CMakeFiles/engine.dir/engine.cc.o.provides

CMakeFiles/engine.dir/engine.cc.o.provides.build: CMakeFiles/engine.dir/engine.cc.o


CMakeFiles/engine.dir/easy_image.cc.o: CMakeFiles/engine.dir/flags.make
CMakeFiles/engine.dir/easy_image.cc.o: ../easy_image.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/atisha/Computer Graphics/engine_old/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/engine.dir/easy_image.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/engine.dir/easy_image.cc.o -c "/home/atisha/Computer Graphics/engine_old/easy_image.cc"

CMakeFiles/engine.dir/easy_image.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/engine.dir/easy_image.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/atisha/Computer Graphics/engine_old/easy_image.cc" > CMakeFiles/engine.dir/easy_image.cc.i

CMakeFiles/engine.dir/easy_image.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/engine.dir/easy_image.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/atisha/Computer Graphics/engine_old/easy_image.cc" -o CMakeFiles/engine.dir/easy_image.cc.s

CMakeFiles/engine.dir/easy_image.cc.o.requires:

.PHONY : CMakeFiles/engine.dir/easy_image.cc.o.requires

CMakeFiles/engine.dir/easy_image.cc.o.provides: CMakeFiles/engine.dir/easy_image.cc.o.requires
	$(MAKE) -f CMakeFiles/engine.dir/build.make CMakeFiles/engine.dir/easy_image.cc.o.provides.build
.PHONY : CMakeFiles/engine.dir/easy_image.cc.o.provides

CMakeFiles/engine.dir/easy_image.cc.o.provides.build: CMakeFiles/engine.dir/easy_image.cc.o


CMakeFiles/engine.dir/l_parser.cc.o: CMakeFiles/engine.dir/flags.make
CMakeFiles/engine.dir/l_parser.cc.o: ../l_parser.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/atisha/Computer Graphics/engine_old/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/engine.dir/l_parser.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/engine.dir/l_parser.cc.o -c "/home/atisha/Computer Graphics/engine_old/l_parser.cc"

CMakeFiles/engine.dir/l_parser.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/engine.dir/l_parser.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/atisha/Computer Graphics/engine_old/l_parser.cc" > CMakeFiles/engine.dir/l_parser.cc.i

CMakeFiles/engine.dir/l_parser.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/engine.dir/l_parser.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/atisha/Computer Graphics/engine_old/l_parser.cc" -o CMakeFiles/engine.dir/l_parser.cc.s

CMakeFiles/engine.dir/l_parser.cc.o.requires:

.PHONY : CMakeFiles/engine.dir/l_parser.cc.o.requires

CMakeFiles/engine.dir/l_parser.cc.o.provides: CMakeFiles/engine.dir/l_parser.cc.o.requires
	$(MAKE) -f CMakeFiles/engine.dir/build.make CMakeFiles/engine.dir/l_parser.cc.o.provides.build
.PHONY : CMakeFiles/engine.dir/l_parser.cc.o.provides

CMakeFiles/engine.dir/l_parser.cc.o.provides.build: CMakeFiles/engine.dir/l_parser.cc.o


CMakeFiles/engine.dir/ini_configuration.cc.o: CMakeFiles/engine.dir/flags.make
CMakeFiles/engine.dir/ini_configuration.cc.o: ../ini_configuration.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/atisha/Computer Graphics/engine_old/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/engine.dir/ini_configuration.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/engine.dir/ini_configuration.cc.o -c "/home/atisha/Computer Graphics/engine_old/ini_configuration.cc"

CMakeFiles/engine.dir/ini_configuration.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/engine.dir/ini_configuration.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/atisha/Computer Graphics/engine_old/ini_configuration.cc" > CMakeFiles/engine.dir/ini_configuration.cc.i

CMakeFiles/engine.dir/ini_configuration.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/engine.dir/ini_configuration.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/atisha/Computer Graphics/engine_old/ini_configuration.cc" -o CMakeFiles/engine.dir/ini_configuration.cc.s

CMakeFiles/engine.dir/ini_configuration.cc.o.requires:

.PHONY : CMakeFiles/engine.dir/ini_configuration.cc.o.requires

CMakeFiles/engine.dir/ini_configuration.cc.o.provides: CMakeFiles/engine.dir/ini_configuration.cc.o.requires
	$(MAKE) -f CMakeFiles/engine.dir/build.make CMakeFiles/engine.dir/ini_configuration.cc.o.provides.build
.PHONY : CMakeFiles/engine.dir/ini_configuration.cc.o.provides

CMakeFiles/engine.dir/ini_configuration.cc.o.provides.build: CMakeFiles/engine.dir/ini_configuration.cc.o


CMakeFiles/engine.dir/vector.cc.o: CMakeFiles/engine.dir/flags.make
CMakeFiles/engine.dir/vector.cc.o: ../vector.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/atisha/Computer Graphics/engine_old/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/engine.dir/vector.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/engine.dir/vector.cc.o -c "/home/atisha/Computer Graphics/engine_old/vector.cc"

CMakeFiles/engine.dir/vector.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/engine.dir/vector.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/atisha/Computer Graphics/engine_old/vector.cc" > CMakeFiles/engine.dir/vector.cc.i

CMakeFiles/engine.dir/vector.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/engine.dir/vector.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/atisha/Computer Graphics/engine_old/vector.cc" -o CMakeFiles/engine.dir/vector.cc.s

CMakeFiles/engine.dir/vector.cc.o.requires:

.PHONY : CMakeFiles/engine.dir/vector.cc.o.requires

CMakeFiles/engine.dir/vector.cc.o.provides: CMakeFiles/engine.dir/vector.cc.o.requires
	$(MAKE) -f CMakeFiles/engine.dir/build.make CMakeFiles/engine.dir/vector.cc.o.provides.build
.PHONY : CMakeFiles/engine.dir/vector.cc.o.provides

CMakeFiles/engine.dir/vector.cc.o.provides.build: CMakeFiles/engine.dir/vector.cc.o


# Object files for target engine
engine_OBJECTS = \
"CMakeFiles/engine.dir/engine.cc.o" \
"CMakeFiles/engine.dir/easy_image.cc.o" \
"CMakeFiles/engine.dir/l_parser.cc.o" \
"CMakeFiles/engine.dir/ini_configuration.cc.o" \
"CMakeFiles/engine.dir/vector.cc.o"

# External object files for target engine
engine_EXTERNAL_OBJECTS =

engine: CMakeFiles/engine.dir/engine.cc.o
engine: CMakeFiles/engine.dir/easy_image.cc.o
engine: CMakeFiles/engine.dir/l_parser.cc.o
engine: CMakeFiles/engine.dir/ini_configuration.cc.o
engine: CMakeFiles/engine.dir/vector.cc.o
engine: CMakeFiles/engine.dir/build.make
engine: CMakeFiles/engine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/atisha/Computer Graphics/engine_old/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable engine"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/engine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/engine.dir/build: engine

.PHONY : CMakeFiles/engine.dir/build

CMakeFiles/engine.dir/requires: CMakeFiles/engine.dir/engine.cc.o.requires
CMakeFiles/engine.dir/requires: CMakeFiles/engine.dir/easy_image.cc.o.requires
CMakeFiles/engine.dir/requires: CMakeFiles/engine.dir/l_parser.cc.o.requires
CMakeFiles/engine.dir/requires: CMakeFiles/engine.dir/ini_configuration.cc.o.requires
CMakeFiles/engine.dir/requires: CMakeFiles/engine.dir/vector.cc.o.requires

.PHONY : CMakeFiles/engine.dir/requires

CMakeFiles/engine.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/engine.dir/cmake_clean.cmake
.PHONY : CMakeFiles/engine.dir/clean

CMakeFiles/engine.dir/depend:
	cd "/home/atisha/Computer Graphics/engine_old/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/atisha/Computer Graphics/engine_old" "/home/atisha/Computer Graphics/engine_old" "/home/atisha/Computer Graphics/engine_old/cmake-build-debug" "/home/atisha/Computer Graphics/engine_old/cmake-build-debug" "/home/atisha/Computer Graphics/engine_old/cmake-build-debug/CMakeFiles/engine.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/engine.dir/depend
