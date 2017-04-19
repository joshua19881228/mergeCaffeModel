#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
using std::ifstream;
using std::vector;

//config file:
//dst_prototxt, dst_model
//src_prototxt_0 src_model_0 flag_auto n pair_0(src:dst) pair_1
//src_prototxt_1 src_prototxt_1 flag_auto n pair_0 pair_1

struct SrcDstPair
{
    string src_layer_name;
    string dst_layer_name;
};

struct SrcInfo
{
    bool flag_auto;    //if true: auto fill weights when src and dst layer names are same 
    string prototxt_path;   //src prototxt path
    string weights_path;    //src weigts path
    vector<SrcDstPair> pairs;   //src layer name - dst layer name
};

struct ConfigInfo
{
    string dst_prototxt_path; //wanted prototxt path
    string dst_model_path;  //output weights path

    vector<SrcInfo> src_info;
};

bool splitPair(const string input, SrcDstPair& a_pair);
bool splitSrcLine(const string input, SrcInfo& a_src_info);
bool loadConfigFile(const string config_file_path, ConfigInfo& config_info);