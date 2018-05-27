
use std;
use CudaHandleHolder;
use CloneStructure;
use GetParams;
use VectorUtils;
use error_calculator::*;
use magnitude_manager::*;


struct TrainingTrainable<T: GetParams + CloneStructure> {
    error: f32,
    trainable: T,
}

struct ChildsRatios {
    ratios: Vec<f32>,
}
impl ChildsRatios {

    fn new(nb_childs: usize) -> ChildsRatios {
        let (a, b) = ChildsRatios::getfnab(nb_childs);
        ChildsRatios { ratios: (0..nb_childs).map(|i| { a*(i as f32)+b }).collect::<Vec<_>>() }
    }

    fn getfnab(n: usize) -> (f32,f32) {
        // a*n + b = 0
        // b*n + a*sum(0,n) = 1
        let mut sum_0n = 0;
        for i in 1..n { sum_0n += i; }
        // a*n + b - b - a*sum(0,n)/n = -1/n
        // a * (n + sum(0,n)/n) = -1/n
        // a = - n / (n*n + sum(0,n))
        let n = n as f32;
        let sum_0n = sum_0n as f32;
        let a = - 1.0 / (n*n - sum_0n);
        let b = - a * n;
        (a,b)
    }

}

pub fn train<T: GetParams + CloneStructure>(cuda: &mut CudaHandleHolder, input_trainable: &mut T,
                           error_calculation: &mut ErrorCalculator<T>, initial_magnitude: f32,
                           nb_childs: usize, depth: usize, nb_iterations: usize, print_every: usize) {

    //let magnitude_manager = LinearMagnitudeManager::new(nb_iterations, initial_magnitude, 0.0);
    let mut magnitude_manager = SkiMagnitudeManagerV2::new(initial_magnitude, 10, 4, 3);

    let mut cesure_parent = input_trainable.clone_structure();
    cesure_parent.params_mut().clone_from_device(input_trainable.params());
    let mut parent_error = error_calculation.compute_error(&cesure_parent, cuda);

    let mut cesure_childs = (0..nb_childs).map(|_| {
        let network = input_trainable.clone_structure();
        TrainingTrainable { trainable: network, error: std::f32::NAN, }
    }).collect::<Vec<_>>();
    let childs_ratio = ChildsRatios::new(nb_childs);

    let mut cesure_tmp = input_trainable.clone_structure();

    for iteration in 0..nb_iterations {

        magnitude_manager.update(parent_error);
        let magnitude = magnitude_manager.get_magnitude();

        let mut sorted_childs_i = Vec::with_capacity(nb_childs);

        for child_i in 0..nb_childs {
            {
                let child = &mut cesure_childs[child_i];
                child.trainable.params_mut().clone_randomized_from(cuda, cesure_parent.params(), magnitude/100.0);
                child.error = error_calculation.compute_error(&child.trainable, cuda);

                for _ in 0..depth {
                    cesure_tmp.params_mut().clone_randomized_from(cuda, child.trainable.params(), magnitude);
                    let tmp_error = error_calculation.compute_error(&cesure_tmp, cuda);
                    if tmp_error <= child.error {
                        std::mem::swap(&mut cesure_tmp, &mut child.trainable);
                        child.error = tmp_error;
                        if tmp_error <= parent_error {
                            break;
                        }
                    }
                }
            }

            for i in 0..nb_childs {
                if i >= sorted_childs_i.len() {
                    sorted_childs_i.push(child_i);
                    break
                } else if cesure_childs[child_i].error < cesure_childs[sorted_childs_i[i]].error {
                    sorted_childs_i.insert(i, child_i);
                    break
                }
            }
        }

        let mut change_pack = sorted_childs_i.iter().map(|i| { cesure_childs[*i].trainable.params() }).collect::<Vec<_>>();
        cesure_parent.params_mut().clone_weighted_from(cuda, change_pack.as_slice(), childs_ratio.ratios.as_slice());
        parent_error = error_calculation.compute_error(&cesure_parent, cuda);

        if iteration % print_every == 0 {
            println!("Iteration {}, Error = {}, Magnitude = {}", iteration, parent_error, magnitude);
        }

    }

    input_trainable.params_mut().clone_from_device(cesure_parent.params());
}