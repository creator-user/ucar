# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ucar/Desktop/ucar/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ucar/Desktop/ucar/build

# Utility rule file for _run_tests_tf_nosetests.

# Include the progress variables for this target.
include geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/progress.make

_run_tests_tf_nosetests: geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/build.make

.PHONY : _run_tests_tf_nosetests

# Rule to build all files generated by this target.
geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/build: _run_tests_tf_nosetests

.PHONY : geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/build

geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/clean:
	cd /home/ucar/Desktop/ucar/build/geometry/tf && $(CMAKE_COMMAND) -P CMakeFiles/_run_tests_tf_nosetests.dir/cmake_clean.cmake
.PHONY : geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/clean

geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/depend:
	cd /home/ucar/Desktop/ucar/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ucar/Desktop/ucar/src /home/ucar/Desktop/ucar/src/geometry/tf /home/ucar/Desktop/ucar/build /home/ucar/Desktop/ucar/build/geometry/tf /home/ucar/Desktop/ucar/build/geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : geometry/tf/CMakeFiles/_run_tests_tf_nosetests.dir/depend

