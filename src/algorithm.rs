use std::cmp::Reverse;
use std::time::Instant;

use kdam::{tqdm, BarExt};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::cache::Cache;
use crate::individual::Individual;

fn initial_population(
    genome_size: usize,
    population_size: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    (0..population_size)
        .map(|_| Individual::new_random(genome_size, rng))
        .collect()
}

fn generate_offspring(population: &[Individual], rng: &mut impl Rng) -> Vec<Individual> {
    let mut new_population = Vec::with_capacity(population.len());

    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.shuffle(rng);

    for chunk in indices.chunks(2) {
        let i = chunk[0];
        let j = chunk[1];
        let parent1 = &population[i];
        let parent2 = &population[j];

        let (child1, child2) = two_point_crossover(parent1, parent2, rng);
        new_population.push(child1);
        new_population.push(child2);
    }

    debug_assert_eq!(new_population.len(), population.len());
    new_population
}

fn two_point_crossover(
    parent1: &Individual,
    parent2: &Individual,
    rng: &mut impl Rng,
) -> (Individual, Individual) {
    parent1.two_point_crossover(parent2, rng)
}

fn mutate(individual: &mut Individual, rng: &mut impl Rng) {
    individual.mutate(rng)
}

fn evaluate(individual: &mut Individual) {
    individual.evaluate();
}

fn evaluate_cached(individual: &mut Individual, cache: &mut Cache) {
    let bitstring = individual.bitstring();
    if let Some(cached) = cache.get(&bitstring) {
        // Overwrite the stored score:
        individual.set_summary(*cached);
    } else {
        let score = individual.evaluate();
        // Update global cache:
        cache.insert(bitstring, score);
    }
    // debug_assert!(individual.score.is_some());
}

fn bitstring(individual: &Individual) -> String {
    individual.bitstring()
}

fn fitness(individual: &Individual) -> f64 {
    individual.summary().count as f64
}

fn diversity(individual: &Individual, population: &[Individual]) -> f64 {
    population
        .iter()
        .map(|other| (fitness(individual) - fitness(other)).abs())
        .sum()
}

pub fn run(
    genome_size: usize,
    population_size: usize,
    num_generations: usize,
    seed: u64,
    is_plus: bool,
) -> std::io::Result<()> {
    let start_time = Instant::now();

    let use_file_cache = envmnt::is("USE_FILE_CACHE");
    println!("USE_FILE_CACHE: {}", use_file_cache);

    // Create a new cache or load an existing one:
    let mut cache = if use_file_cache {
        match Cache::load_from_file("cache.bin") {
            Ok(cache) => {
                println!("Loaded cache with {} entries", cache.data.len());
                cache
            }
            Err(e) => {
                println!("Error loading the cache: {}", e);
                Cache::new()
            }
        }
    } else {
        Cache::new()
    };

    // Determine the max fitness:
    let max_fitness = genome_size as f64;

    // Seeded random generator:
    let mut rng = StdRng::seed_from_u64(seed);

    // Create an initial population:
    let mut population = initial_population(genome_size, population_size, &mut rng);

    // Evaluate the initial population:
    for individual in population.iter_mut() {
        // evaluate_cached(individual, &mut cache);
        evaluate(individual);
    }

    // Sort the initial population in descending order (best first):
    population.sort_by_key(|x| Reverse(OrderedFloat(fitness(x))));

    let mut best = population[0].clone();
    let mut best_generation = 0;

    println!("Initial population has size {}", population.len());
    println!(
        "Best initial individual with fitness {}: {}",
        fitness(&best),
        best.bitstring()
    );

    // Check if the stopping condition is met:
    if (fitness(&population[0]) - max_fitness).abs() < f64::EPSILON {
        println!("[!] Initial population already has max fitness");
    }

    println!("Running EA for {} generations...", num_generations);
    let mut pb = tqdm!(total = num_generations);

    for generation in 1..=num_generations {
        // Inner loop:
        //   - produce offspring:
        //      - select parents
        //      - crossover parents
        //      - produce children
        //      - mutate children
        //   - evaluate offspring
        //   - merge population with offspring (replace or extend)
        //   - select new population
        //   - (update best)
        //
        // Invariant: in the beginning of each generation,
        //            the population is evaluated and sorted.

        // Create an offspring:
        let mut offspring = generate_offspring(&population, &mut rng);

        // Mutate the offspring:
        for individual in offspring.iter_mut() {
            mutate(individual, &mut rng);
        }

        // Evaluate the offspring:
        for individual in offspring.iter_mut() {
            // evaluate_cached(individual, &mut cache);
            evaluate(individual);
        }

        // Update the population:
        if is_plus {
            // (\mu + \lambda)
            population.extend(offspring);
        } else {
            // (\mu, \lambda)
            population = offspring;
        }

        // Sort the population in descending order (best first):
        let scores = population
            .iter()
            .map(|this| {
                let fitness = fitness(this);
                let diversity = diversity(this, &population);
                let score = fitness + 0.1 * diversity;
                score
            })
            .collect::<Vec<_>>();
        let mut scored = population.into_iter().zip(scores).collect::<Vec<_>>();
        scored.sort_by_key(|(_, s)| Reverse(OrderedFloat(*s)));
        population = scored.into_iter().map(|(x, _)| x).collect();

        // Select best individuals:
        population.truncate(population_size);

        // Update the best found individual:
        if fitness(&population[0]) > fitness(&best) {
            best = population[0].clone();
            best_generation = generation;
        }

        // Progress:
        // pb.set_description(format!("GEN {}", generation));
        pb.update(1)?;

        if generation <= 10
            || (generation < 100 && generation % 10 == 0)
            || (generation < 1000 && generation % 100 == 0)
            || (generation < 10000 && generation % 1000 == 0)
            || (generation % 10000 == 0)
        {
            pb.write(format!(
                "[{}/{}] Best individual has fitness {}, diversity {}: {}",
                generation,
                num_generations,
                fitness(&population[0]),
                diversity(&population[0], &population),
                population[0].bitstring()
            ))?;
        }

        // Check if the stopping condition is met:
        if (fitness(&best) - max_fitness).abs() < f64::EPSILON {
            pb.write(format!("Reached max fitness on generation {}", generation))?;
            break;
        }
    }

    pb.refresh()?;
    eprintln!();

    println!("\n-----------------\n");
    println!(
        "Best individual from generation {} with fitness {}: {}",
        best_generation,
        fitness(&best),
        best.bitstring()
    );

    // Save the updated cache to the file:
    if use_file_cache {
        println!("Saving cache with {} entries", cache.data.len());
        cache.save_to_file("cache.bin").unwrap();
    }

    let elapsed = Instant::now() - start_time;
    println!(
        "All done in {}.{:03} s",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    Ok(())
}
