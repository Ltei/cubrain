
use cumath::{CuVector, DEFAULT_STREAM};
use super::ErrorCalculator;
use training::training_set::PackedTrainingSet;
use CudaHandleHolder;
use GetParams;
use ForwardInference;


pub struct ErrorCalculatorSimple<'a> {
    training_set: &'a PackedTrainingSet,
    workspace: CuVector<f32>,
    output_buffer: CuVector<f32>,
}
impl<'a> ErrorCalculatorSimple<'a> {
    pub fn new(computation: &ForwardInference, training_set: &'a PackedTrainingSet) -> ErrorCalculatorSimple<'a> {
        ErrorCalculatorSimple {
            training_set,
            workspace: CuVector::<f32>::zero(computation.workspace_len()),
            output_buffer: CuVector::<f32>::zero(training_set.packed_outputs.len()),
        }
    }
}
impl<'a, T: GetParams + ForwardInference> ErrorCalculator<T> for ErrorCalculatorSimple<'a> {
    fn compute_error(&mut self, computation: &T, cuda: &mut CudaHandleHolder) -> f32 {
        {
            let mut iter = self.output_buffer.slice_mut_iter();
            for input in &self.training_set.inputs {
                computation.forward_inference(cuda, input, &mut iter.next(computation.output_len()).unwrap(), &mut self.workspace);
            }
            assert_eq!(iter.len(), 0);
        }

        self.output_buffer.sub(&self.training_set.packed_outputs, &DEFAULT_STREAM);
        cuda.cublas.asum(&self.output_buffer)
    }
}