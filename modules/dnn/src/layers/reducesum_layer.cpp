// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

namespace cv { namespace dnn {

int axis = 0;
class ReduceSumLayerImpl CV_FINAL : public ReduceSumLayer
{
public:
    ReduceSumLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        
        if (!params.has("axes")){
            axis = -1;
        }
        else{

            DictValue axes = params.get("axes");
            CV_Assert(axes.size() <=1 );
            if(axes.getIntValue(0)==0) axis = 0;
            if(axes.getIntValue(0)==1) axis = 1;
        }
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        //CV_Assert(inputs.empty());
        //std::cout<<"input size "<<inputs.size()<<std::endl;
        ///std::cout<<"blob shape "<<shape(blobs[0])<<std::endl;  /////blob size is zero
        //std::cout<<"input size "<<inputs[0].size()<<std::endl;

        // for(int i=0;i<inputs.size();i++){
        //     for(int j=0;j<inputs[i].size();j++){
        //         std::cout<<inputs[i][j]<<" ";
        //     }
        //     std::cout<<" "<<std::endl;
        // }
        if(axis==-1){
            std::vector<int> outShape(1);    
            outShape[0] = 1;    
            outputs.assign(1, outShape);
        }
        else if(axis==0){
            std::vector<int> outShape(2);
            outShape[0] = inputs[0][1];
            outShape[1] = 1;
            //std::cout<<"outShape[1] "<<outShape[1]<<" inputs[0][1] "<<inputs[0][1]<<std::endl;
            outputs.assign(1, outShape);
        }
        else if(axis==1){
            std::vector<int> outShape(2);
            outShape[0] = inputs[0][0];
            outShape[1] = 1;
            outputs.assign(1, outShape);
        }
        return true;
    }

// #ifdef HAVE_OPENCL
//     bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
//     {
//         std::vector<UMat> outputs;
//         outs.getUMatVector(outputs);
//         if (outs.depth() == CV_16S)
//             convertFp16(blobs[0], outputs[0]);
//         else
//             blobs[0].copyTo(outputs[0]);
//         return true;
//     }
// #endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<cv::Mat> inputs, outputs;
        
        inputs_arr.getMatVector(inputs);
        
        outputs_arr.getMatVector(outputs);

        //std::cout<<"input size "<<inputs.size()<<" rows "<<inputs[0].rows<<" cols "<<inputs[0].cols<<std::endl;
        //std::cout<<"outputs size "<<outputs.size()<<" rows "<<outputs[0].rows<<" cols "<<outputs[0].cols<<std::endl;

        if(axis == -1){

            for(int i=0;i<inputs.size();i++){
            //std::cout<<"input "<<i<<std::endl;
            int sum = 0;
            for(int j=0;j<inputs[i].rows;j++){
                //std::cout<<"input row "<<j<<std::endl;
                for(int k=0;k<inputs[i].cols;k++){
                    //std::cout<<inputs[i].at<float>(j,k)<<" ";
                    sum+= inputs[i].at<float>(j,k);
                    }
                //std::cout<<" "<<std::endl;
                }
            outputs[i].at<float>(0,0) = sum; 
            //std::cout<<i<<" sum "<<outputs[i].at<float>(0,0)<<std::endl;
            }
        }

        if(axis == 0){

            for(int i=0;i<inputs.size();i++){
                //cv::Mat sum_col = cv::Mat::zeros(cv::Size(inputs[i].cols,1), CV_32FC1);
                // std::cout<<"input "<<i<<" rows "<<inputs[i].rows<<std::endl;
                // std::cout<<"input "<<i<<" cols "<<inputs[i].cols<<std::endl;
                // std::cout<<"sum_col "<<i<<" cols "<<sum_col.cols<<std::endl;
                // std::cout<<"sum_col "<<i<<" rows "<<sum_col.rows<<std::endl;

                for(int j=0;j<inputs[i].cols;j++){
                    int sum = 0;
                    for(int k=0;k<inputs[i].rows;k++){

                        //std::cout<<inputs[i].at<float>(k,j)<<" ";
                        sum+= inputs[i].at<float>(k,j);
                        
                        }       
                    // std::cout<<" "<<std::endl;
                    // std::cout<<" j "<<j<<std::endl;
                    outputs[i].at<float>(j,0) = sum;
                    }
                //std::cout<<i<<" sum "<<outputs[i].at<float>(0,0)<<std::endl;
                //outputs[i] = sum_col; 
            }

        }

        if(axis == 1){

            for(int i=0;i<inputs.size();i++){
                //cv::Mat sum_row = cv::Mat::zeros(cv::Size(1,inputs[i].rows), CV_32FC1);
                for(int j=0;j<inputs[i].rows;j++){
                    int sum = 0;
                    for(int k=0;k<inputs[i].cols;k++){

                        //std::cout<<inputs[i].at<float>(j,k)<<" ";
                        sum+= inputs[i].at<float>(j,k);
                        
                        }       
                    //std::cout<<" "<<std::endl;
                    outputs[i].at<float>(j,0) = sum;
                    }
                //std::cout<<i<<" sum "<<outputs[i].at<float>(0,0)<<std::endl;
                //outputs[i] = sum_row; 
            }
            
        }

        //std::cout<<"outputs size "<<outputs.size()<<" rows "<<outputs[0].rows<<" cols "<<outputs[0].cols<<std::endl;

    }
};

Ptr<Layer> ReduceSumLayer::create(const LayerParams& params)
{
    return Ptr<Layer>(new ReduceSumLayerImpl(params));
}

}}  // namespace cv::dnn
