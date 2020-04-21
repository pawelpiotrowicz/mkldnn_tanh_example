
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "dnnl.hpp"
#include "example_utils.hpp"
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;
void eltwise_example(dnnl::engine::kind engine_kind) {
    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);
    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
    // Tensor dimensions.
   /*
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 227, // tensor height
            IW = 227; // tensor width
    */
 const memory::dim N = 2, // batch size
            IC = 2, // channels
            IH = 2, // tensor height
            IW = 2; // tensor width
   

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims dst_dims = {N, IC, IH, IW};
    // Allocate buffers. In this example, out-of-place primitive execution is
    // demonstrated since both src and dst are required for later backward
    // propagation.
   
   /*
    std::vector<float> src_data(product(src_dims));
    std::vector<float> dst_data(product(dst_dims));
   */
    std::vector<float> src_data(N*IC*IH*IW);
    std::vector<float> dst_data(N*IC*IH*IW);
   
   
    // Initialize src tensor.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    // Create src and dst memory descriptors and memory objects.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);
    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);
    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);
    // Create operation descriptor.
   /*
    auto eltwise_d = eltwise_forward::desc(prop_kind::forward_training,
            algorithm::eltwise_relu, src_md, 0.f, 0.f);
   */
    auto eltwise_d = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_tanh, src_md, 0.f, 0.f);
   
    // Create primitive descriptor.
    auto eltwise_pd = eltwise_forward::primitive_desc(eltwise_d, engine);
    // Create the primitive.
    auto eltwise_prim = eltwise_forward(eltwise_pd);
    // Primitive arguments.
    std::unordered_map<int, memory> eltwise_args;
    eltwise_args.insert({DNNL_ARG_SRC, src_mem});
    eltwise_args.insert({DNNL_ARG_DST, dst_mem});
    // Primitive execution: element-wise (ReLU).
    eltwise_prim.execute(engine_stream, eltwise_args);
    // Wait for the computation to finalize.
    engine_stream.wait();
    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);

    for(size_t i=0;i<dst_data.size();++i)
    {
         auto f1 = dst_data[i];
         auto f2 = std::tanh(src_data[i]);
       //  std::cout << f1 << " == " << f2  << std::endl;
        //  if(f1!=f2)
        //  {
        //      printf (" %.20f   %.20f \n", f1 , f2);
        //      std::cout << "ERROR diff found " << f1 << " " << f2 << std::endl;
        //     // break;
        //  } else {
        //       std::cout << "OK :)" << std::endl;
        //  }
         printf("%.20f\n",f1);

        
    }

    
     


}


int main(int argc, char **argv) {
   auto k =  parse_engine_kind(argc, argv);
   eltwise_example(k);
   //  return handle_example_errors(
     //       eltwise_example, parse_engine_kind(argc, argv));
   
   //  eltwise_example(eltwise_example);

 //  return 

}