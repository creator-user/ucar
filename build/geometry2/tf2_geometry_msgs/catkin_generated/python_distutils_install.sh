#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/ucar/Desktop/ucar/src/geometry2/tf2_geometry_msgs"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/ucar/Desktop/ucar/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/ucar/Desktop/ucar/install/lib/python3/dist-packages:/home/ucar/Desktop/ucar/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/ucar/Desktop/ucar/build" \
    "/usr/bin/python3" \
    "/home/ucar/Desktop/ucar/src/geometry2/tf2_geometry_msgs/setup.py" \
    build --build-base "/home/ucar/Desktop/ucar/build/geometry2/tf2_geometry_msgs" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/home/ucar/Desktop/ucar/install" --install-scripts="/home/ucar/Desktop/ucar/install/bin"
