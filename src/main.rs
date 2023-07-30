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
struct Args {
    /// Genome size
    #[arg(short, long, default_value_t = 10)]
    genome_size: usize,

    /// Population size
    #[arg(short, long, default_value_t = 10)]
    population_size: usize,

    /// Number of generations
    #[arg(short, long, default_value_t = 1000)]
    num_generations: usize,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    kdam::term::init(false);
    kdam::term::hide_cursor()?;

    let args = Args::parse();

    evolution::algorithm::run(
        args.genome_size,
        args.population_size,
        args.num_generations,
        false,
    )?;

    Ok(())
}
