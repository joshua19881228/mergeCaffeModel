# mergeCaffeModel #

A tool is implemented to merge caffemodels according to a config file. The source code can be found in this [repo](https://github.com/joshua19881228/mergeCaffeModel)

Under development...

## Dependency ##

Any version of Caffe that has been successfully compiled.

## Usage ##

A configuration file should be prepared. The file content should be like

    [path to dst prototxt] [path to dst caffemodel]
    [path to src prototxt] [path to src caffemodel] [auto_flag (0 or 1)] [num_pairs(int)] [src_layer_name_(1):dst_layer_name_(1)] ... [src_layer_name_(num_pairs):dst_layer_name_(num_pairs)]
    [path to src prototxt] [path to src caffemodel] [auto_flag (0 or 1)] [num_pairs(int)] [src_layer_name_(1):dst_layer_name_(1)] ... [src_layer_name_(num_pairs):dst_layer_name_(num_pairs)]
    ...

The first line gives the path to wanted caffemodel's prototxt and the path to save the wanted caffemodel.

From the second line, source information are given. The first two strings gives the path to the source caffemodel's prototxt and the path to source caffemodel. Then a variable (0 or 1) is used to indicate whether to automatically copy weights when the src and dst layer names are same. (1 true, others false). The forth parameter tells how many pairs of src and dst layer would be processed. From then on, pairs are give with formant of "src\_layer\_name:dst\_layer\_name"

An example can be found below

    dst.prototxt dst.caffemodel
    src0.prototxt src0.caffemodel 1 0
    src1.prototxt src1.caffemodel 0 1 layer_0:layer_dst_0

It means that all the layers in dst.caffemodel that share the same name with those in src0.caffemodel will be filled by the weights in src0.caffemodel as long as they have the exact same number of parameters. And the layer_dst_0 in dst.caffemodel will be filled by the weigths of layer_0 in src1.caffemodel.

An example code is as follows

    #include "mergeModel.h"

    int main()
    {
        string config_file_path = "test_config.txt";
        MergeModelClass test_class(config_file_path);
        test_class.mergeModel();

        return 0;
    }

Of course you can write your own main function, which can take inputs from command line.

**NOTE** only float type is supported.

## 20170420 ##

A class is defined to merge caffemodels.

Now the project can copy src weights to dst weights automatically if the two layers have same name. Further test will be carried out to confirm weights can be copied from src caffemodel using pairs.

## 20170423 ##

Copying weights using pairs has been conformed. 