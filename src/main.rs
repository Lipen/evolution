use clap::Parser;

/// Evolutionary algorithms playground
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
#[command(help_template = "\
{before-help}{name} {version}
{author-with-newline}{about-section}
{usage-heading} {usage}

{all-args}{after-help}
")]
struct Cli {
    /// Genome size
    #[arg(short, long, default_value_t = 10)]
    genome_size: usize,

    /// Population size
    #[arg(short, long, default_value_t = 10)]
    population_size: usize,

    /// Number of generations
    #[arg(short, long, default_value_t = 1000)]
    num_generations: usize,

    /// Seed for PRNG
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Is plus?
    #[arg(long)]
    plus: bool,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    kdam::term::init(false);
    kdam::term::hide_cursor()?;

    let cli = Cli::parse();

    assert_eq!(
        cli.population_size % 2,
        0,
        "Population size must be an even number"
    );

    evolution::algorithm::run(
        cli.genome_size,
        cli.population_size,
        cli.num_generations,
        cli.seed,
        cli.plus,
    )?;

    Ok(())
}
