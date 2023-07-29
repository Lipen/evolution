use std::cmp::{Ordering, Reverse};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::Instant;

use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const GENOME_SIZE: usize = 20;
const IS_PLUS: bool = true; // (1+1) vs (1,1)
const POPULATION_SIZE: usize = 10; // \mu
const NUM_GENERATIONS: usize = 10000;
const MAX_FITNESS: f64 = GENOME_SIZE as f64;

#[derive(Debug)]
struct Individual {
    genome: Vec<bool>,
    pub score: Option<Score>,
}

impl Individual {
    pub fn new(genome: Vec<bool>) -> Self {
        Self {
            genome,
            score: None,
        }
    }

    pub fn new_random(size: usize, rng: &mut impl Rng) -> Self {
        let genome = (0..size).map(|_| rng.gen()).collect();
        Self::new(genome)
    }
}

impl Clone for Individual {
    fn clone(&self) -> Self {
        Self::new(self.genome.clone())
    }
}

impl Individual {
    pub fn bitstring(&self) -> String {
        self.genome
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }

    pub fn compute_score(&mut self, cache: &mut Cache) {
        if let Some(cached) = cache.get(&self.genome) {
            // Overwrite the stored score:
            self.score = Some(*cached);
        } else {
            let score = *self.score.get_or_insert_with(|| {
                // Compute the fitness here:
                let count = self.genome.iter().filter(|x| **x).count();
                let fitness = count as f64;
                Score { fitness, count }
            });
            // Update global cache:
            cache.insert(self.genome.clone(), score);
        }
    }

    pub fn mutate(&mut self, rng: &mut impl Rng) {
        let n = self.genome.len();
        let p = 1.0 / (n as f64);
        for i in 0..n {
            if rng.gen::<f64>() < p {
                self.genome[i] = !self.genome[i];
            }
        }
    }

    pub fn two_point_crossover(
        &self,
        other: &Individual,
        rng: &mut impl Rng,
    ) -> (Individual, Individual) {
        debug_assert_eq!(self.genome.len(), GENOME_SIZE);
        debug_assert_eq!(other.genome.len(), GENOME_SIZE);

        let left = rng.gen_range(0..GENOME_SIZE);
        let right = rng.gen_range(left..=GENOME_SIZE);

        let mut child1 = self.clone();
        let mut child2 = other.clone();

        for i in left..right {
            std::mem::swap(&mut child1.genome[i], &mut child2.genome[i]);
        }

        (child1, child2)
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct Score {
    pub fitness: f64,
    pub count: usize,
}

impl Score {
    fn key(&self) -> (OrderedFloat<f64>, usize) {
        (OrderedFloat(self.fitness), self.count)
    }
}

impl Eq for Score {}

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        self.key().eq(&other.key())
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Serialize, Deserialize)]
struct Cache {
    data: HashMap<Vec<bool>, Score>,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    // Load the cache from a file
    pub fn load_from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(file_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let config = bincode::config::standard();
        let (cache, _) = bincode::serde::decode_from_slice(&buffer, config)?;
        Ok(cache)
    }

    // Save the cache into a file
    pub fn save_to_file(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let config = bincode::config::standard();
        let serialized = bincode::serde::encode_to_vec(self, config)?;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)?;
        file.write_all(&serialized)?;
        Ok(())
    }
}

impl Cache {
    pub fn get(&self, key: &Vec<bool>) -> Option<&Score> {
        self.data.get(key)
    }

    pub fn insert(&mut self, key: Vec<bool>, value: Score) {
        self.data.insert(key, value);
    }
}

fn compute_score(individual: &mut Individual, cache: &mut Cache) {
    individual.compute_score(cache)
}

fn evaluate_all(population: &mut [Individual], cache: &mut Cache) {
    // Evaluate the new population
    for individual in population.iter_mut() {
        compute_score(individual, cache);
    }

    // Sort the population by fitness (and count, for equal fitness) in descending order (max first):
    population.sort_by_key(|x| {
        let score = x.score.unwrap();
        let f = OrderedFloat(score.fitness);
        let c = score.count;
        Reverse((f, c))
    });
}

fn two_point_crossover(
    parent1: &Individual,
    parent2: &Individual,
    rng: &mut impl Rng,
) -> (Individual, Individual) {
    parent1.two_point_crossover(parent2, rng)
}

fn initial_population(rng: &mut impl Rng) -> Vec<Individual> {
    (0..POPULATION_SIZE)
        .map(|_| Individual::new_random(GENOME_SIZE, rng))
        .collect()
}

fn generate_offspring(population: &[Individual], rng: &mut impl Rng) -> Vec<Individual> {
    debug_assert_eq!(population.len(), POPULATION_SIZE);

    let mut new_population = Vec::with_capacity(POPULATION_SIZE);

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

    debug_assert_eq!(new_population.len(), POPULATION_SIZE);
    new_population
}

fn mutate(individual: &mut Individual, rng: &mut impl Rng) {
    individual.mutate(rng)
}

pub fn run() {
    let start_time = Instant::now();

    let use_file_cache = std::env::var("USE_FILE_CACHE").is_ok();
    println!("USE_FILE_CACHE: {}", use_file_cache);

    // Create a new cache or load an existing one:
    let mut cache = if use_file_cache {
        match Cache::load_from_file("cache.bin") {
            Ok(cache) => {
                println!("Loaded cache with {} entries", cache.data.len());
                cache
            }
            Err(_) => Cache::new(),
        }
    } else {
        Cache::new()
    };

    // Seeded random generator:
    let mut rng = StdRng::seed_from_u64(42);

    // Create an initial population:
    let mut population = initial_population(&mut rng);

    // Evaluate the initial population:
    for individual in population.iter_mut() {
        compute_score(individual, &mut cache);
    }

    // Sort the initial population:
    population.sort_by_key(|x| Reverse(x.score.unwrap()));

    println!("Initial generation of size {}:", population.len());
    for individual in population.iter() {
        println!(
            "  - {} (fitness = {}, count = {})",
            individual.bitstring(),
            individual.score.unwrap().fitness,
            individual.score.unwrap().count
        );
    }
    println!(
        "[0/{}] Best individual has fitness {}: {}",
        NUM_GENERATIONS,
        population[0].score.unwrap().fitness,
        population[0].bitstring()
    );

    let mut best = population[0].clone();
    let mut best_score = population[0].score.unwrap();
    let mut best_generation = 0;

    // Check if the stopping condition is met:
    if (population[0].score.unwrap().fitness - MAX_FITNESS).abs() < f64::EPSILON {
        println!("[!] Initial population already has max fitness");
    }

    for generation in 1..=NUM_GENERATIONS {
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
            compute_score(individual, &mut cache);
        }

        // Update the population:
        if IS_PLUS {
            // (\mu + \lambda)
            population.extend(offspring);
        } else {
            // (\mu, \lambda)
            population = offspring;
        }

        // Sort the population by fitness (and count, for equal fitness) in descending order (max first):
        population.sort_by_key(|x| Reverse(x.score.unwrap()));

        // Select best individuals:
        population.truncate(POPULATION_SIZE);

        // Update the best found individual:
        if population[0].score.unwrap().fitness > best_score.fitness {
            best = population[0].clone();
            best_score = population[0].score.unwrap();
            best_generation = generation;
        }

        // Report:
        if generation <= 10
            || (generation < 100 && generation % 10 == 0)
            || (generation < 1000 && generation % 100 == 0)
            || (generation % 1000 == 0)
        {
            println!(
                "[{}/{}] Best individual has fitness {}: {}",
                generation,
                NUM_GENERATIONS,
                population[0].score.unwrap().fitness,
                population[0].bitstring()
            );
        }

        // Check if the stopping condition is met:
        if (population[0].score.unwrap().fitness - MAX_FITNESS).abs() < f64::EPSILON {
            println!("Reached max fitness on generation {}", generation);
            break;
        }
    }

    println!("\n-----------------\n");
    println!(
        "Best individual from generation {} with fitness {}: {}",
        best_generation,
        best_score.fitness,
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
    )
}
