"""Command-line interface for DeepBrief."""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()
app = typer.Typer(help="DeepBrief - Video Analysis Application")


@app.command()
def analyze(
    video_path: Optional[Path] = typer.Argument(None, help="Path to video file to analyze"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for reports"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """
    Analyze a video file for presentation feedback.
    
    If no video path is provided, launches the web interface.
    """
    console.print(Panel.fit(
        "[bold blue]DeepBrief[/bold blue]\n"
        "[dim]Video Analysis Application[/dim]",
        border_style="blue"
    ))
    
    if video_path:
        # CLI mode - analyze specific video
        console.print(f"[green]Analyzing video:[/green] {video_path}")
        if output_dir:
            console.print(f"[blue]Output directory:[/blue] {output_dir}")
        if config_file:
            console.print(f"[blue]Config file:[/blue] {config_file}")
        
        # TODO: Implement CLI analysis mode
        console.print("[yellow]CLI analysis mode not yet implemented. Use web interface for now.[/yellow]")
    else:
        # Web UI mode
        console.print("[green]Launching web interface...[/green]")
        
        # TODO: Import and launch Gradio interface
        console.print("[yellow]Web interface not yet implemented.[/yellow]")
        console.print("[dim]Run with --help for available options.[/dim]")


@app.command()
def version() -> None:
    """Show version information."""
    from deep_brief import __version__
    console.print(f"DeepBrief version {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()