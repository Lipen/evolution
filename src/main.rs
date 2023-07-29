fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    kdam::term::init(false);
    kdam::term::hide_cursor()?;

    evolution::algorithm::run()?;

    Ok(())
}
