
use cumath::*;

use CesureNetwork;
use network::*;


pub fn compute_memory_usage_score(cesure: &CesureGRU, cublas: &Cublas, workspace: &mut Workspace, context_buffer: &mut [&mut CuVector]) -> f32 {

    let mut score_sum = 0.0;
    let mut score_count = 0;

    cesure.new_sequence(&CuVector::new(cesure.info_dimension(), 0.5), workspace);

    for i in 0..context_buffer.len() {
        cesure.compute_next(cublas, workspace);
        for last_i in (0..i).rev() {
            let (last_context,context) = context_buffer.split_at_mut(i);
            CuVector::sub(&workspace.buffer.slice(cesure.info_dimension(), cesure.context_dimension()), last_context[last_i], context[0], &DEFAULT_STREAM);
            score_sum += cublas.asum(context[0]);
            score_count += 1;
        }
        context_buffer[i].clone_from_device(&workspace.buffer.slice(cesure.info_dimension(), cesure.context_dimension()));
    }

    score_sum / (cesure.context_dimension() * score_count) as f32

}

pub fn train(cublas: &Cublas, curand: &mut CurandGenerator, cesure: &mut CesureGRU,
             magnitude0: f32, magnitude1: f32, iterations: usize, print_every: usize) {

    let mut workspace = cesure.create_workspace();
    let mut context_buffer: Vec<CuVector> = (0..25).map(|_| { CuVector::new(cesure.context_dimension, 0.0) }).collect();
    let mut context_buffer: Vec<&mut CuVector> = context_buffer.iter_mut().map(|x| { x }).collect();

    let mut best_cesure = cesure.clone_structure();
    best_cesure.clone_from(&cesure);
    let mut best_score = compute_memory_usage_score(&best_cesure, cublas, &mut workspace, context_buffer.as_mut_slice());

    let mut tmp_cesure = cesure.clone_structure();

    for iteration in 0..iterations {

        let x = iteration as f32 / iterations as f32;
        let magnitude = x * magnitude1 + (1.0-x) * magnitude0;

        tmp_cesure.clone_randomized_from(curand, &best_cesure, magnitude);
        let tmp_score = compute_memory_usage_score(&tmp_cesure, cublas, &mut workspace, context_buffer.as_mut_slice());

        if tmp_score >= best_score {
            best_cesure.clone_from(&tmp_cesure);
            best_score = tmp_score;
        }

        if iteration % print_every == 0 {
            println!("Iteration {}, Score = {}", iteration, best_score);
        }

    }

    cesure.clone_from(&best_cesure);

}