
use super::*;
use cumath_nn::*;



#[derive(Clone, Debug)]
pub enum LayerDescriptor {
    Dense { descriptor: DenseLayerDescriptor },
    Convolution { descriptor: ConvolutionLayerDescriptor },
    Activation { descriptor: ActivationLayerDescriptor },
}
impl LayerDescriptor {
    pub fn dense(input_len: usize, output_len: usize) -> LayerDescriptor {
        LayerDescriptor::Dense {
            descriptor: DenseLayerDescriptor{input_len, output_len}
        }
    }
    pub fn convolution(input_rows: usize, input_cols: usize, kernels: Vec<ConvolutionKernelDescriptor>) -> LayerDescriptor {
        let mut output_len = 0;
        let mut weights_count = 0;
        let kernels = kernels.iter().map(|x| {
            let r = input_rows - x.rows;
            let c = input_cols - x.cols;
            assert_eq!(r % x.row_step as usize, 0);
            assert_eq!(c % x.col_step as usize, 0);
            let output_rows = r / x.row_step as usize + 1;
            let output_cols = c / x.col_step as usize + 1;
            output_len += output_rows * output_cols;
            weights_count += x.rows * x.cols;
            ConvolutionKernelInfo {
                rows: x.rows,         cols: x.cols,
                row_step: x.row_step, col_step: x.col_step,
                output_rows,          output_cols,
            }
        }).collect();
        LayerDescriptor::Convolution {
            descriptor: ConvolutionLayerDescriptor{input_rows, input_cols, kernels, output_len, weights_count}
        }
    }
    pub fn activation(mode: CudnnActivationMode, len: usize, coef: f64) -> LayerDescriptor {
        LayerDescriptor::Activation {
            descriptor: ActivationLayerDescriptor{mode, len, coef}
        }
    }

    pub fn input_len(&self) -> usize {
        match self {
            LayerDescriptor::Dense {descriptor} => {
                descriptor.input_len
            },
            LayerDescriptor::Convolution {descriptor} => {
                descriptor.input_rows * descriptor.input_cols
            },
            LayerDescriptor::Activation {descriptor} => {
                descriptor.len
            },
        }
    }
    pub fn output_len(&self) -> usize {
        match self {
            LayerDescriptor::Dense {descriptor} => {
                descriptor.output_len
            },
            LayerDescriptor::Convolution {descriptor} => {
                descriptor.output_len
            },
            LayerDescriptor::Activation {descriptor} => {
                descriptor.len
            },
        }
    }
    pub fn weights_count(&self) -> usize {
        match self {
            LayerDescriptor::Dense {descriptor} => {
                descriptor.input_len * descriptor.output_len
            },
            LayerDescriptor::Convolution {descriptor} => {
                descriptor.weights_count
            },
            LayerDescriptor::Activation {..} => {
                0
            },
        }
    }

    pub fn duplicate(&self) -> LayerDescriptor {
        match self {
            LayerDescriptor::Dense {descriptor} => {
                LayerDescriptor::Dense {
                    descriptor: descriptor.clone()
                }
            },
            LayerDescriptor::Convolution {descriptor} => {
                LayerDescriptor::Convolution {
                    descriptor: descriptor.clone()
                }
            },
            LayerDescriptor::Activation {descriptor} => {
                LayerDescriptor::Activation {
                    descriptor: descriptor.clone()
                }
            },
        }
    }
    pub fn create_layer(&self, data: CuVectorPtr<f32>) -> Box<Layer> {
        match self {
            LayerDescriptor::Dense {descriptor} => {
                descriptor.create_layer(data)
            },
            LayerDescriptor::Convolution {descriptor} => {
                descriptor.create_layer(data)
            },
            LayerDescriptor::Activation {descriptor} => {
                descriptor.create_layer(data)
            },
        }
    }
}