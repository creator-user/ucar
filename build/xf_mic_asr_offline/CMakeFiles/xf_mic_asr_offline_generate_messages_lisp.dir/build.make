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

# Utility rule file for xf_mic_asr_offline_generate_messages_lisp.

# Include the progress variables for this target.
include xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/progress.make

xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/msg/Pcm_Msg.lisp
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Offline_Result_srv.lisp
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Major_Mic_srv.lisp
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Major_Mic_srv.lisp
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Start_Record_srv.lisp
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Awake_Word_srv.lisp
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Led_On_srv.lisp
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.lisp


/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/msg/Pcm_Msg.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/msg/Pcm_Msg.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg/Pcm_Msg.msg
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating Lisp code from xf_mic_asr_offline/Pcm_Msg.msg"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg/Pcm_Msg.msg -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/msg

/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Offline_Result_srv.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Offline_Result_srv.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Offline_Result_srv.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating Lisp code from xf_mic_asr_offline/Get_Offline_Result_srv.srv"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Offline_Result_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv

/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Major_Mic_srv.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Major_Mic_srv.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Major_Mic_srv.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Generating Lisp code from xf_mic_asr_offline/Set_Major_Mic_srv.srv"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Major_Mic_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv

/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Major_Mic_srv.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Major_Mic_srv.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Major_Mic_srv.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Generating Lisp code from xf_mic_asr_offline/Get_Major_Mic_srv.srv"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Major_Mic_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv

/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Start_Record_srv.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Start_Record_srv.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Start_Record_srv.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Generating Lisp code from xf_mic_asr_offline/Start_Record_srv.srv"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Start_Record_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv

/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Awake_Word_srv.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Awake_Word_srv.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Awake_Word_srv.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Generating Lisp code from xf_mic_asr_offline/Set_Awake_Word_srv.srv"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Awake_Word_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv

/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Led_On_srv.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Led_On_srv.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Led_On_srv.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Generating Lisp code from xf_mic_asr_offline/Set_Led_On_srv.srv"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Set_Led_On_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv

/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.lisp: /opt/ros/melodic/lib/genlisp/gen_lisp.py
/home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.lisp: /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ucar/Desktop/ucar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Generating Lisp code from xf_mic_asr_offline/Get_Awake_Angle_srv.srv"
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && ../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/melodic/share/genlisp/cmake/../../../lib/genlisp/gen_lisp.py /home/ucar/Desktop/ucar/src/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.srv -Ixf_mic_asr_offline:/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg -Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg -p xf_mic_asr_offline -o /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv

xf_mic_asr_offline_generate_messages_lisp: xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/msg/Pcm_Msg.lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Offline_Result_srv.lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Major_Mic_srv.lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Major_Mic_srv.lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Start_Record_srv.lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Awake_Word_srv.lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Set_Led_On_srv.lisp
xf_mic_asr_offline_generate_messages_lisp: /home/ucar/Desktop/ucar/devel/share/common-lisp/ros/xf_mic_asr_offline/srv/Get_Awake_Angle_srv.lisp
xf_mic_asr_offline_generate_messages_lisp: xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/build.make

.PHONY : xf_mic_asr_offline_generate_messages_lisp

# Rule to build all files generated by this target.
xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/build: xf_mic_asr_offline_generate_messages_lisp

.PHONY : xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/build

xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/clean:
	cd /home/ucar/Desktop/ucar/build/xf_mic_asr_offline && $(CMAKE_COMMAND) -P CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/cmake_clean.cmake
.PHONY : xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/clean

xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/depend:
	cd /home/ucar/Desktop/ucar/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ucar/Desktop/ucar/src /home/ucar/Desktop/ucar/src/xf_mic_asr_offline /home/ucar/Desktop/ucar/build /home/ucar/Desktop/ucar/build/xf_mic_asr_offline /home/ucar/Desktop/ucar/build/xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : xf_mic_asr_offline/CMakeFiles/xf_mic_asr_offline_generate_messages_lisp.dir/depend

