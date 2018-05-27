
use cumath::*;

#[derive(Clone, Copy)]
pub struct ActivationFunction {
    function: fn(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>),
    function_: fn(input: &mut CuVectorDeref<f32>),
    function_deriv: fn(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>),
    function_deriv_: fn(input: &mut CuVectorDeref<f32>),
}

impl ActivationFunction {
    pub fn identity() -> ActivationFunction {
        ActivationFunction {
            function: identity,
            function_: identity_,
            function_deriv: identity_deriv,
            function_deriv_: identity_deriv_
        }
    }
    pub fn sigmoid() -> ActivationFunction {
        ActivationFunction {
            function: sigmoid,
            function_: sigmoid_,
            function_deriv: sigmoid_deriv,
            function_deriv_: sigmoid_deriv_,
        }
    }
    pub fn tanh() -> ActivationFunction {
        ActivationFunction { 
            function: tanh, 
            function_: tanh_,
            function_deriv: tanh_deriv,
            function_deriv_: tanh_deriv_,
        }
    }
    pub fn relu() -> ActivationFunction {
        ActivationFunction {
            function: relu,
            function_: relu_,
            function_deriv: relu_deriv,
            function_deriv_: relu_deriv_,
        }
    }


    pub fn activate(&self, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) { (self.function)(input, output) }
    pub fn activate_(&self, input: &mut CuVectorDeref<f32>) { (self.function_)(input) }

    pub fn derivate(&self, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) { (self.function_deriv)(input, output) }
    pub fn derivate_(&self, input: &mut CuVectorDeref<f32>) { (self.function_deriv_)(input) }
}

fn identity(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
    output.clone_from_device(input)
}
fn identity_(_input: &mut CuVectorDeref<f32>) {
    
}
fn identity_deriv(_input: &CuVectorDeref<f32>, _output: &mut CuVectorDeref<f32>) {
    _output.init(1.0, &DEFAULT_STREAM)
}
fn identity_deriv_(input: &mut CuVectorDeref<f32>) {
    input.init(1.0, &DEFAULT_STREAM)
}

fn sigmoid(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
    CuVectorMath::<f32>::sigmoid(input, output, &DEFAULT_STREAM) 
}
fn sigmoid_(input: &mut CuVectorDeref<f32>) {
    input.fast_sigmoid(&DEFAULT_STREAM)
}
fn sigmoid_deriv(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
    CuVectorMath::<f32>::sigmoid_deriv(input, output, &DEFAULT_STREAM)
}
fn sigmoid_deriv_(input: &mut CuVectorDeref<f32>) {
    input.fast_sigmoid_deriv(&DEFAULT_STREAM)
}

fn tanh(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
    CuVectorMath::<f32>::tanh(input, output, &DEFAULT_STREAM) 
}
fn tanh_(input: &mut CuVectorDeref<f32>) {
    input.fast_tanh(&DEFAULT_STREAM)
}
fn tanh_deriv(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
    CuVectorMath::<f32>::tanh_deriv(input, output, &DEFAULT_STREAM)
}
fn tanh_deriv_(input: &mut CuVectorDeref<f32>) {
    input.fast_tanh_deriv(&DEFAULT_STREAM)
}

fn relu(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
    CuVectorMath::<f32>::relu(input, output, &DEFAULT_STREAM)
}
fn relu_(input: &mut CuVectorDeref<f32>) {
    input.relu(&DEFAULT_STREAM)
}
fn relu_deriv(input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
    CuVectorMath::<f32>::relu_deriv(input, output, &DEFAULT_STREAM)
}
fn relu_deriv_(input: &mut CuVectorDeref<f32>) {
    input.relu_deriv(&DEFAULT_STREAM)
}