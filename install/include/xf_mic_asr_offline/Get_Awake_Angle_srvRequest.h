// Generated by gencpp from file xf_mic_asr_offline/Get_Awake_Angle_srvRequest.msg
// DO NOT EDIT!


#ifndef XF_MIC_ASR_OFFLINE_MESSAGE_GET_AWAKE_ANGLE_SRVREQUEST_H
#define XF_MIC_ASR_OFFLINE_MESSAGE_GET_AWAKE_ANGLE_SRVREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace xf_mic_asr_offline
{
template <class ContainerAllocator>
struct Get_Awake_Angle_srvRequest_
{
  typedef Get_Awake_Angle_srvRequest_<ContainerAllocator> Type;

  Get_Awake_Angle_srvRequest_()
    {
    }
  Get_Awake_Angle_srvRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> const> ConstPtr;

}; // struct Get_Awake_Angle_srvRequest_

typedef ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<std::allocator<void> > Get_Awake_Angle_srvRequest;

typedef boost::shared_ptr< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest > Get_Awake_Angle_srvRequestPtr;
typedef boost::shared_ptr< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest const> Get_Awake_Angle_srvRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace xf_mic_asr_offline

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsMessage': True, 'IsFixedSize': True, 'HasHeader': False}
// {'xf_mic_asr_offline': ['/home/ucar/Desktop/ucar/src/xf_mic_asr_offline/msg'], 'std_msgs': ['/opt/ros/melodic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsMessage< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "xf_mic_asr_offline/Get_Awake_Angle_srvRequest";
  }

  static const char* value(const ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
;
  }

  static const char* value(const ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Get_Awake_Angle_srvRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::xf_mic_asr_offline::Get_Awake_Angle_srvRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // XF_MIC_ASR_OFFLINE_MESSAGE_GET_AWAKE_ANGLE_SRVREQUEST_H
