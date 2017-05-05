#include "caffe/mergeModel.h"
int main(int argc, char* argv[])
{
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;
	if(argc!=2)
	{
		LOG(ERROR)<<"Usage: mergeMode [path to config file]";
		return 1;
	}
	string config_file_path = argv[1];
	MergeModelClass merge_model_class(config_file_path);
	merge_model_class.mergeModel();
	return 0;
}
