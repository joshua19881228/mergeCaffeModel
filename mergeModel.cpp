#include "mergeModel.h"
#include <strstream>
using std::strstream;

bool splitPair(const string input, SrcDstPair& a_pair)
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

bool splitSrcLine(const string input, SrcInfo& a_src_info)
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

bool loadConfigFile(const string config_file_path, ConfigInfo& config_info)
{
    std::ifstream config_file(config_file_path.c_str());
    int set_num = 0;
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