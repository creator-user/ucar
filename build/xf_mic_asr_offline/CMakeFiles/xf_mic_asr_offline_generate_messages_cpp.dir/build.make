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


# Produce verbose output by default.
VERBOSE = 1

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

# Utility rule file for xf_mic_asr_offline_generate_messages_cpp.

# Include the progress variables for this target.
include xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/progress.make

xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Pcm_Msg.h
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Offline_Result_srv.h
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Major_Mic_srv.h
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Major_Mic_srv.h
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Start_Record_srv.h
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Awake_Word_srv.h
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Led_On_srv.h
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Awake_Angle_srv.h


/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Pcm_Msg.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Pcm_Msg.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg/Pcm_Msg.msg
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Pcm_Msg.h: /opt/ros/melodic/share/gencpp/msg.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating C++ code from xf_mic_asr_offline/Pcm_Msg.msg"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg/Pcm_Msg.msg -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Offline_Result_srv.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Offline_Result_srv.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Offline_Result_srv.srv
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Offline_Result_srv.h: /opt/ros/melodic/share/gencpp/msg.h.template
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Offline_Result_srv.h: /opt/ros/melodic/share/gencpp/srv.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating C++ code from xf_mic_asr_offline/Get_Offline_Result_srv.srv"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Offline_Result_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Major_Mic_srv.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Major_Mic_srv.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Major_Mic_srv.srv
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Major_Mic_srv.h: /opt/ros/melodic/share/gencpp/msg.h.template
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Major_Mic_srv.h: /opt/ros/melodic/share/gencpp/srv.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Generating C++ code from xf_mic_asr_offline/Set_Major_Mic_srv.srv"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Major_Mic_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Major_Mic_srv.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Major_Mic_srv.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Major_Mic_srv.srv
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Major_Mic_srv.h: /opt/ros/melodic/share/gencpp/msg.h.template
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Major_Mic_srv.h: /opt/ros/melodic/share/gencpp/srv.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Generating C++ code from xf_mic_asr_offline/Get_Major_Mic_srv.srv"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Major_Mic_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Start_Record_srv.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Start_Record_srv.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Start_Record_srv.srv
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Start_Record_srv.h: /opt/ros/melodic/share/gencpp/msg.h.template
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Start_Record_srv.h: /opt/ros/melodic/share/gencpp/srv.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Generating C++ code from xf_mic_asr_offline/Start_Record_srv.srv"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Start_Record_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Awake_Word_srv.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Awake_Word_srv.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Awake_Word_srv.srv
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Awake_Word_srv.h: /opt/ros/melodic/share/gencpp/msg.h.template
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Awake_Word_srv.h: /opt/ros/melodic/share/gencpp/srv.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Generating C++ code from xf_mic_asr_offline/Set_Awake_Word_srv.srv"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Awake_Word_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Led_On_srv.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Led_On_srv.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Led_On_srv.srv
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Led_On_srv.h: /opt/ros/melodic/share/gencpp/msg.h.template
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Led_On_srv.h: /opt/ros/melodic/share/gencpp/srv.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Generating C++ code from xf_mic_asr_offline/Set_Led_On_srv.srv"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Led_On_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Awake_Angle_srv.h: /opt/ros/melodic/lib/gencpp/gen_cpp.py
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Awake_Angle_srv.h: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.srv
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Awake_Angle_srv.h: /opt/ros/melodic/share/gencpp/msg.h.template
/home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Awake_Angle_srv.h: /opt/ros/melodic/share/gencpp/srv.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Generating C++ code from xf_mic_asr_offline/Get_Awake_Angle_srv.srv"
	cd /home/ucar/Desktop/ucar/src/xf_mic_asr_offline && /home/ucar/Desktop/ucar/build/catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline -e /opt/ros/melodic/share/gencpp/cmake/..

xf_mic_asr_offline_generate_messages_cpp: xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Pcm_Msg.h
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Offline_Result_srv.h
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Major_Mic_srv.h
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Major_Mic_srv.h
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Start_Record_srv.h
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Awake_Word_srv.h
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Set_Led_On_srv.h
xf_mic_asr_offline_generate_messages_cpp: /home/ucar/Desktop/ucar/devel/include/xf_mic_asr_offline/Get_Awake_Angle_srv.h
xf_mic_asr_offline_generate_messages_cpp: xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/build.make

.PHONY : xf_mic_asr_offline_generate_messages_cpp

# Rule to build all files generated by this target.
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/build: xf_mic_asr_offline_generate_messages_cpp

.PHONY : xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/build

xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/clean:
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && $(CMAKE_COMMAND) -P CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/cmake_clean.cmake
.PHONY : xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/clean

xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/depend:
	cd /home/ucar/Desktop/ucar/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ucar/Desktop/ucar/src /home/ucar/Desktop/ucar/src/xf_mic_asr_offline /home/ucar/Desktop/ucar/build /home/ucar/Desktop/ucar/build/xf_mic_asr_offline /home/ucar/Desktop/ucar/build/xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_cpp.dir/depend

