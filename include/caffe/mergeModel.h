#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layer.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::NetParameter;
using std::string;
using std::ifstream;
using std::vector;
using boost::shared_ptr;

//config file:
//dst_prototxt, dst_model
//src_prototxt_0 src_model_0 flag_auto n pair_0(src:dst) pair_1
//src_prototxt_1 src_prototxt_1 flag_auto n pair_0 pair_1

typedef struct SrcDstPair
{
    string src_layer_name;
    string dst_layer_name;
};

typedef struct SrcInfo
{
    bool flag_auto;    //if true: auto fill weights when src and dst layer names are same 
    string prototxt_path;   //src prototxt path
    string weights_path;    //src weigts path
    vector<SrcDstPair> pairs;   //src layer name - dst layer name
};

typedef struct ConfigInfo
{
    string dst_prototxt_path; //wanted prototxt path
    string dst_model_path;  //output weights path

    vector<SrcInfo> src_info;
};

class MergeModelClass
{
public:
    MergeModelClass(const string config_file_path)
    { 
        dst_net = NULL;
        if(loadConfigFile(config_file_path))
            dst_net_init = initDstModel();
    };
    ~MergeModelClass()
    {
        if (dst_net_init)
            delete dst_net;
    };
    
    bool mergeModel();
    

private:
    bool loadConfigFile(const string config_file_path);

    bool splitPair(const string input, SrcDstPair& a_pair);
    bool splitSrcLine(const string input, SrcInfo& a_src_info);
    bool initDstModel();
    bool initSrcModel(const string prototxt_path, const string weights_path, Net<float>** net);
    bool copyWeights(Net<float>* src_net, vector<SrcDstPair> pairs, bool auto_flag);
    bool copyWeights(Net<float>* src_net, const string src_layer_name, const string dst_layer_name);
    void addAutoCopyLayers(Net<float>* src_net, vector<SrcDstPair>& pairs);

    Net<float>* dst_net;
    ConfigInfo config_info;

    bool dst_net_init;
};
