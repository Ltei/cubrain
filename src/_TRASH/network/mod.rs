
mod layer_dense;
pub use self::layer_dense::*;

mod sigmoid;
pub use self::sigmoid::*;

mod network;
pub use self::network::*;

mod computation;
pub use self::computation::*;

mod computation_descriptor;
pub(crate) use self::computation_descriptor::*;


use std::convert::From;
use cumath::{CuVector, CuVectorMutPtr, CuVectorDeref, DEFAULT_STREAM};
use meta::Save;
use CubrainResult;
use CudaHandleHolder;


pub(crate) enum ComputationParams {
    Owned(CuVector<f32>),
    Borrowed(CuVectorMutPtr<f32>),
}