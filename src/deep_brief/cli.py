"""Command-line interface for DeepBrief."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from deep_brief.utils.config import get_config, load_config

console = Console()
app = typer.Typer(help="DeepBrief - Video Analysis Application")


@app.command()
def analyze(
    video_path: Path | None = typer.Argument(
        None, help="Path to video file to analyze"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory for reports"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    _verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """
    Analyze a video file for presentation feedback.

    If no video path is provided, launches the web interface.
    """
    # Load configuration
    config = load_config(config_file) if config_file else get_config()

    # Set up logging
    logger = logging.getLogger("deep_brief")
    logger.info("Starting DeepBrief analysis")

    console.print(
        Panel.fit(
            f"[bold blue]{config.app_name}[/bold blue]\n[dim]Video Analysis Application[/dim]",
            border_style="blue",
        )
    )

    if config.debug:
        console.print("[yellow]Debug mode enabled[/yellow]")
        logger.debug("Debug mode is active")

    if video_path:
        # CLI mode - analyze specific video
        logger.info(f"Analyzing video: {video_path}")
        console.print(f"[green]Analyzing video:[/green] {video_path}")

        if output_dir:
            console.print(f"[blue]Output directory:[/blue] {output_dir}")
            logger.debug(f"Output directory: {output_dir}")

        if config_file:
            console.print(f"[blue]Config file:[/blue] {config_file}")
            logger.debug(f"Using config file: {config_file}")

        # Show relevant config info in debug mode
        if config.debug:
            console.print(
                f"[dim]Max file size: {config.processing.max_video_size_mb}MB[/dim]"
            )
            console.print(
                f"[dim]Transcription model: {config.transcription.model}[/dim]"
            )

        # TODO: Implement CLI analysis mode
        logger.warning("CLI analysis mode not yet implemented")
        console.print(
            "[yellow]CLI analysis mode not yet implemented. Use web interface for now.[/yellow]"
        )
    else:
        # Web UI mode
        logger.info("Launching web interface")
        console.print("[green]Launching web interface...[/green]")

        # TODO: Import and launch Gradio interface
        logger.warning("Web interface not yet implemented")
        console.print("[yellow]Web interface not yet implemented.[/yellow]")
        console.print("[dim]Run with --help for available options.[/dim]")


@app.command()
def version() -> None:
    """Show version information."""
    from deep_brief import __version__

    console.print(f"DeepBrief version {__version__}")


@app.command()
def config(
    show_all: bool = typer.Option(
        False, "--all", help="Show all configuration options"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
) -> None:
    """Show current configuration."""
    if config_file:
        config_obj = load_config(config_file)
        console.print(f"[blue]Using config file:[/blue] {config_file}")
    else:
        config_obj = get_config()
        console.print("[blue]Using default configuration[/blue]")

    console.print("\n[bold]Application Configuration[/bold]")
    console.print(f"App Name: {config_obj.app_name}")
    console.print(f"Version: {config_obj.version}")
    console.print(f"Debug Mode: {config_obj.debug}")

    if show_all:
        console.print("\n[bold]Processing Settings[/bold]")
        console.print(f"Max Video Size: {config_obj.processing.max_video_size_mb}MB")
        console.print(
            f"Supported Formats: {', '.join(config_obj.processing.supported_formats)}"
        )
        console.print(f"Temp Directory: {config_obj.processing.temp_dir}")

        console.print("\n[bold]Transcription Settings[/bold]")
        console.print(f"Model: {config_obj.transcription.model}")
        console.print(f"Language: {config_obj.transcription.language}")
        console.print(f"Device: {config_obj.transcription.device}")

        console.print("\n[bold]Analysis Settings[/bold]")
        console.print(f"Target WPM: {config_obj.analysis.target_wpm_range}")
        console.print(
            f"Confidence Threshold: {config_obj.analysis.confidence_threshold}"
        )

        console.print("\n[bold]Logging Settings[/bold]")
        console.print(f"Level: {config_obj.logging.level}")
        console.print(f"File: {config_obj.logging.file_path}")
    else:
        console.print("\n[dim]Use --all to see all configuration options[/dim]")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
