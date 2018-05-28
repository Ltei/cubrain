
use super::*;
use cumath::*;
use CudaHandleHolder;



pub struct ConvolutionLayer {
    pub(super) input_rows: usize,
    pub(super) input_cols: usize,
    pub(super) kernels: Vec<ConvolutionKernel>,
    pub(super) output_len: usize,
    pub(super) weights_count: usize,
}
pub(super) struct ConvolutionKernel {
    pub(super) kernel: CuMatrixPtr<f32>,
    pub(super) output_rows: usize,
    pub(super) output_cols: usize,
    pub(super) row_step: usize,
    pub(super) col_step: usize,
}


impl ForwardInference for ConvolutionLayer {

    fn input_len(&self) -> usize { self.input_rows * self.input_cols }
    fn output_len(&self) -> usize { self.output_len }
    fn workspace_len(&self) -> usize { 0 }

    fn forward_inference(&self, _cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, _workspace: &mut CuVectorDeref<f32>) {
        let mut output = output.slice_mut_iter();
        for kernel in &self.kernels {
            CuMatrixMath::<f32>::convolution(&input.matrix_slice(0, self.input_rows, self.input_cols), unsafe { kernel.kernel.deref() },
                                             &mut output.next_matrix(kernel.output_rows, kernel.output_cols).unwrap(),
                                             kernel.row_step as i32, kernel.col_step as i32, &DEFAULT_STREAM);
        }
    }

}

impl Layer for ConvolutionLayer {
    fn clone_structure(&self, data: CuVectorPtr<f32>) -> Box<Layer> {
        let mut iter = unsafe { data.deref().slice_iter() };
        let output = Box::new(ConvolutionLayer {
            input_rows: self.input_rows,
            input_cols: self.input_cols,
            kernels: self.kernels.iter().map(|x| {
                ConvolutionKernel {
                    kernel: iter.next_matrix(x.kernel.rows(), x.kernel.cols()).unwrap().as_wrapped_ptr(),
                    output_rows: x.output_rows,
                    output_cols: x.output_cols,
                    row_step: x.row_step,
                    col_step: x.col_step,
                }
            }).collect(),
            output_len: self.output_len,
            weights_count: self.weights_count,
        });
        assert_eq!(iter.len(), 0);
        output
    }

    fn params_count(&self) -> usize {
        self.weights_count
    }

    fn forward_training(&self, _cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
        let mut output = output.slice_mut_iter();
        for kernel in &self.kernels {
            CuMatrixMath::<f32>::convolution(&input.matrix_slice(0, self.input_rows, self.input_cols), unsafe { kernel.kernel.deref() },
                                             &mut output.next_matrix(kernel.output_rows, kernel.output_cols).unwrap(),
                                             kernel.row_step as i32, kernel.col_step as i32, &DEFAULT_STREAM);
        }
    }

    #[allow(unused_variables)]
    fn backward_training(&self, cuda: &mut CudaHandleHolder, learning_rate: f32, momentum: f32,
                     layer_input: &CuVectorDeref<f32>, layer_output: &mut CuVectorDeref<f32>, front_signal: &CuVectorDeref<f32>,
                     weights_change: &mut CuVectorDeref<f32>, back_signal: Option<&mut CuVectorDeref<f32>>) {
        unimplemented!()
        /*assert_eq!(layer_input.len(), self.descriptor.input_len());
        assert_eq!(front_signal.len(), self.descriptor.output_len());
        assert_eq!(weights_change.len(), self.descriptor.weights_count());

        self.descriptor.activation.derivate_(layer_output);
        layer_output.pmult(front_signal, &DEFAULT_STREAM);*/
    }
}


#[derive(Clone, Debug)]
pub struct ConvolutionLayerDescriptor {
    pub(super) input_rows: usize,
    pub(super) input_cols: usize,
    pub(super) kernels: Vec<ConvolutionKernelInfo>,
    pub(super) output_len: usize,
    pub(super) weights_count: usize
}
impl ConvolutionLayerDescriptor {
    pub fn create_layer(&self, data: CuVectorPtr<f32>) -> Box<Layer> {
        let mut iter = unsafe { data.deref().slice_iter() };
        let output = Box::new(ConvolutionLayer {
            input_rows: self.input_rows,
            input_cols: self.input_cols,
            kernels: self.kernels.iter().map(|x| {
                ConvolutionKernel {
                    kernel: iter.next_matrix(x.rows, x.cols).unwrap().as_wrapped_ptr(),
                    output_rows: x.output_rows,
                    output_cols: x.output_cols,
                    row_step: x.row_step,
                    col_step: x.col_step,
                }
            }).collect(),
            output_len: self.output_len,
            weights_count: self.weights_count,
        });
        assert_eq!(iter.len(), 0);
        output
    }
}



#[derive(Clone, Debug)]
pub struct ConvolutionKernelDescriptor {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_step: usize,
    pub(super) col_step: usize,
}
impl ConvolutionKernelDescriptor {
    pub fn new(rows: usize, cols: usize, row_step: usize, col_step: usize) -> ConvolutionKernelDescriptor {
        ConvolutionKernelDescriptor { rows, cols, row_step, col_step }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ConvolutionKernelInfo {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_step: usize,
    pub(super) col_step: usize,
    pub(super) output_rows: usize,
    pub(super) output_cols: usize,
}