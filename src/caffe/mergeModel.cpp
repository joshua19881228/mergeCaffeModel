#include "caffe/mergeModel.h"
#include <strstream>
using std::strstream;

bool MergeModelClass::splitPair(const string input, SrcDstPair& a_pair)
{
    int first_idx = input.find_first_of(":");
    int last_idx = input.find_last_of(":");
    if (first_idx == last_idx && first_idx>0)
    {
        a_pair.src_layer_name = input.substr(0, last_idx);
        a_pair.dst_layer_name = input.substr(last_idx + 1, input.length() - last_idx - 1);
        return true;
    }
    else
    {
        LOG(ERROR) << "Illegal src:dst pair: " << input;
        return false;
    }
}

bool MergeModelClass::splitSrcLine(const string input, SrcInfo& a_src_info)
{
    strstream a_stream;
    vector<string> splits;
    int num_pairs;
    a_stream << input;    
    while (!a_stream.eof())
    {
        string a_part;
        a_stream >> a_part;
        if(a_part.length() >= 1)
            splits.push_back(a_part);
    }

    if (splits.size() <= 3)
    {
        LOG(ERROR) << "illegal src info (src_prototxt src_weights auto_flag num_pair pairs): " << input;
        return false;
    }

    a_stream.clear();
    a_stream << splits[3];
    a_stream >> num_pairs;

    if (splits.size() != num_pairs + 4)
    {
        LOG(ERROR) << "number of pair is not equal to num_pair " << splits.size()-4 << "!=" << num_pairs;
        return false;
    }

    a_src_info.prototxt_path = splits[0];
    a_src_info.weights_path = splits[1];
    a_src_info.flag_auto = splits[2] == "0" ? false : true;
    
    for (int i = 4; i < splits.size(); i++)
    {
        SrcDstPair a_pair;
        splitPair(splits[i], a_pair);
        a_src_info.pairs.push_back(a_pair);
    }
    return true;
}

bool MergeModelClass::loadConfigFile(const string config_file_path)
{
    std::ifstream config_file(config_file_path.c_str());
    if (config_file)
    {
        string first_line;
        getline(config_file, first_line);
        if (first_line.find_first_of(" ") != first_line.find_last_of(" ") || first_line.find_first_of(" ") < 0)
        {
            LOG(ERROR) << "first line should be dst_prototxt_path dst_weights_path: " << first_line;
            return false;
        }
        config_info.dst_prototxt_path = first_line.substr(0, first_line.find(" "));
        config_info.dst_model_path = first_line.substr(first_line.find(" ") + 1, first_line.length() - first_line.find(" ") - 1);
	std::cout<<config_file.eof()<<std::endl;
	while (!config_file.eof())
        {
            string a_line;
            SrcInfo a_src_info;
            getline(config_file, a_line);
            splitSrcLine(a_line, a_src_info);
            config_info.src_info.push_back(a_src_info);
        }
    }
    else
    {
        LOG(ERROR) << "load config file failed:" << config_file_path;
        return false;
    }
    LOG(INFO) << "Config file loaded";
    config_file.close();
    return true;
}

bool MergeModelClass::initDstModel()
{
    dst_net = new Net<float>(config_info.dst_prototxt_path, caffe::TRAIN);
    if (dst_net == NULL)
        return false;
    //FILE* file = fopen(config_info.dst_model_path.c_str(), "r");
    //if (file == NULL)
    //    LOG(WARNING) << "dst model initialized from empty weights";
    //else
    //{
    //    dst_net->CopyTrainedLayersFrom(config_info.dst_model_path);
    //    fclose(file);
    //}
    return true;
}

bool MergeModelClass::initSrcModel(const string prototxt_path, const string weights_path, Net<float>** net)
{
    *net = new Net<float>(prototxt_path, caffe::TRAIN);
    (*net)->CopyTrainedLayersFrom(weights_path);
    return true;
}

bool MergeModelClass::copyWeights(Net<float>* src_net, const string src_layer_name, const string dst_layer_name)
{
    LOG(INFO) << "Copying " << src_layer_name << " to " << dst_layer_name;
    const shared_ptr<Layer<float> > src_layer = src_net->layer_by_name(src_layer_name);
    if (src_layer == NULL)
    {
        LOG(ERROR) << "src net has no " << src_layer_name;
        return false;
    }
    const shared_ptr<Layer<float> > dst_layer = dst_net->layer_by_name(dst_layer_name);
    if (dst_layer == NULL)
    {
        LOG(ERROR) << "dst net has no " << dst_layer_name;
        return false;
    }
    
    if (src_layer->blobs().size() != dst_layer->blobs().size())
    {
        LOG(ERROR) << "src layer " << src_layer_name
            << " and dst layer " << dst_layer_name
            << " have different number of blobs: "
            << src_layer->blobs().size() << " != " << dst_layer->blobs().size();
        return false;
    }

    for (int b = 0; b < src_layer->blobs().size(); b++)
    {
        string src_blob_shape = src_layer->blobs()[b]->shape_string();
        string dst_blob_shape = dst_layer->blobs()[b]->shape_string();
        if (src_blob_shape != dst_blob_shape)
        {
            LOG(ERROR) << "src layer " << src_layer_name
                << " and dst layer " << dst_layer_name
                << " have different shape of blob "
                << src_blob_shape << " != " << dst_blob_shape;
            return false;
        }

        memcpy(dst_layer->blobs()[b]->mutable_cpu_data(),
            src_layer->blobs()[b]->cpu_data(),
            sizeof(float)*(src_layer->blobs()[b]->count()));
    }
    return true;
}

void MergeModelClass::addAutoCopyLayers(Net<float>* src_net, vector<SrcDstPair>& pairs)
{
    for (int l = 0; l < dst_net->layer_names().size(); l++)
    {
        string layer_name = dst_net->layer_names()[l];
        if (src_net->has_layer(layer_name))
        {
            SrcDstPair a_pair;
            a_pair.src_layer_name = layer_name;
            a_pair.dst_layer_name = layer_name;
            pairs.push_back(a_pair);
        }            
    }
}

bool MergeModelClass::copyWeights(Net<float>* src_net, vector<SrcDstPair> pairs, bool auto_flag)
{    
    if (auto_flag)
        addAutoCopyLayers(src_net, pairs);
    for (int i = 0; i < pairs.size(); i++)
    {
        if (!copyWeights(src_net, pairs[i].src_layer_name, pairs[i].dst_layer_name))
            return false;
    }
    return true;
}

bool MergeModelClass::mergeModel()
{
    if (!dst_net_init)
    {
        LOG(ERROR) << "dst_net needs to be initialized first";
        return false;
    }

    for (int i = 0; i < config_info.src_info.size(); i++)
    {
        bool flag_auto = config_info.src_info[i].flag_auto;
        string prototxt_path = config_info.src_info[i].prototxt_path;
        string weights_path = config_info.src_info[i].weights_path;
        Net<float>* src_net = NULL;
        initSrcModel(prototxt_path, weights_path, &src_net);
        LOG(INFO) << "Start to copy weights from " << prototxt_path;
        copyWeights(src_net, config_info.src_info[i].pairs, flag_auto);
    }
    NetParameter net_param;
    dst_net->ToProto(&net_param, false);
    caffe::WriteProtoToBinaryFile(net_param, config_info.dst_model_path);
    return true;
}
