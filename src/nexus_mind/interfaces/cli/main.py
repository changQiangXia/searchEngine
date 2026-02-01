"""NexusMind CLI - Command line interface."""

from __future__ import annotations

from pathlib import Path

import psutil
import typer
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from nexus_mind.core.engine import NexusEngine
from nexus_mind.infrastructure.memory.manager import get_memory_manager

app = typer.Typer(
    name="nexus",
    help="NexusMind - Next-gen multimodal semantic search",
    no_args_is_help=True,
)
console = Console()


def get_engine(workspace: str | None = None) -> NexusEngine:
    """Get or create engine instance."""
    return NexusEngine(workspace_dir=workspace)


@app.command()
def index(
    paths: list[Path] = typer.Argument(..., help="Image files or directories"),  # noqa: B008
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Processing batch size"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Search directories recursively"
    ),
) -> None:
    """Index images for semantic search."""
    engine = get_engine(workspace)

    # Validate paths
    valid_paths: list[str | Path] = []
    for path in paths:
        if path.exists():
            valid_paths.append(path)
        else:
            console.print(f"[yellow]⚠️  Path not found: {path}[/yellow]")

    if not valid_paths:
        console.print("[red]❌ No valid paths provided[/red]")
        raise typer.Exit(1)

    # Index with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Initializing...", total=None)

        try:
            stats = engine.index_images(
                valid_paths,
                batch_size=batch_size,
                recursive=recursive,
            )
        except Exception as e:
            console.print(f"[red]❌ Indexing failed: {e}[/red]")
            raise typer.Exit(1) from None

    # Display results
    console.print(f"\n[green]✅ Indexed {stats['count']} images[/green]")
    console.print(f"   Time: {stats['time_seconds']:.1f}s")
    console.print(f"   Speed: {stats['vectors_per_second']:.1f} images/s")
    console.print(f"   Index type: {stats['index_type']}")
    console.print(f"   GPU: {'Yes' if stats['on_gpu'] else 'No'}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (text or image path)"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    diverse: bool = typer.Option(False, "--diverse", "-d", help="Use MMR for diverse results"),
) -> None:
    """Search the index."""
    engine = get_engine(workspace)

    # Validate index exists
    if engine.index is None:
        console.print("[red]❌ No index found. Run 'nexus index' first.[/red]")
        raise typer.Exit(1)

    # Search
    try:
        if diverse:
            results = engine.search_diverse(query, top_k=top_k)
        else:
            results = engine.search(query, top_k=top_k)
    except Exception as e:
        console.print(f"[red]❌ Search failed: {e}[/red]")
        raise typer.Exit(1) from None

    # Display results
    console.print(f"\n[bold]Search: '{query}'[/bold]")
    console.print(f"Found {len(results)} results\n")

    table = Table(
        title="Search Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", style="cyan", justify="right", width=6)
    table.add_column("Score", style="green", justify="right", width=8)
    table.add_column("Name", style="white", overflow="fold")
    table.add_column("Path", style="dim", overflow="fold")

    for r in results:
        path = r["metadata"].get("path", "N/A")
        name = r["metadata"].get("name", Path(path).name)
        table.add_row(
            str(r["rank"]),
            f"{r['score']:.3f}",
            name,
            str(Path(path).parent),
        )

    console.print(table)


@app.command()
def status(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
) -> None:
    """Show system and engine status."""
    import torch

    console.print("\n[bold]NexusMind System Status[/bold]\n")

    # PyTorch info
    table = Table(box=box.SIMPLE)
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", "Yes" if torch.cuda.is_available() else "No")

    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
        table.add_row("CUDA Version", torch.version.cuda or "Unknown")

    console.print(table)

    # Memory status
    console.print("\n[bold]Memory Status[/bold]\n")

    mem_manager = get_memory_manager()
    stats = mem_manager.get_stats()

    table = Table(box=box.SIMPLE)
    table.add_column("Type", style="cyan")
    table.add_column("Used", style="yellow")
    table.add_column("Total", style="green")
    table.add_column("Usage", style="magenta")

    if torch.cuda.is_available():
        table.add_row(
            "GPU",
            f"{stats.gpu_used / 1e9:.2f} GB",
            f"{stats.gpu_total / 1e9:.2f} GB",
            f"{stats.gpu_usage_pct:.1f}%",
        )

    ram_total = psutil.virtual_memory().total
    ram_used = ram_total - stats.ram_available
    table.add_row(
        "RAM",
        f"{ram_used / 1e9:.2f} GB",
        f"{ram_total / 1e9:.2f} GB",
        f"{ram_used / ram_total * 100:.1f}%",
    )

    console.print(table)

    # Engine status
    console.print("\n[bold]Engine Status[/bold]\n")

    try:
        engine = get_engine(workspace)
        engine_stats = engine.get_stats()

        table = Table(box=box.SIMPLE)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Workspace", engine_stats["workspace"])
        table.add_row("CLIP Model", engine_stats["clip_model"])
        table.add_row("CLIP Device", engine_stats["clip_device"])

        if engine_stats["index"]:
            table.add_row("Index Vectors", str(engine_stats["index"]["vectors"]))
            table.add_row("Index Type", engine_stats["index"]["type"])
            table.add_row("GPU Index", "Yes" if engine_stats["index"]["on_gpu"] else "No")
        else:
            table.add_row("Index", "[red]Not built[/red]")

        console.print(table)

    except Exception as e:
        console.print(f"[yellow]⚠️  Could not load engine: {e}[/yellow]")


@app.command()
def negative(
    positive: str = typer.Argument(..., help="Positive query (what to include)"),
    negative: str = typer.Argument(..., help="Negative query (what to exclude)"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    alpha: float = typer.Option(0.7, "--alpha", "-a", help="Negative weight (0-1)"),
) -> None:
    """Negative search - find images matching positive but not negative."""
    engine = get_engine(workspace)

    if engine.index is None:
        console.print("[red]❌ No index found. Run 'nexus index' first.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Searching: '{positive}' but NOT '{negative}'[/bold]\n")

    try:
        results = engine.negative_search(positive, negative, top_k=top_k, alpha=alpha)
    except Exception as e:
        console.print(f"[red]❌ Search failed: {e}[/red]")
        raise typer.Exit(1) from None

    # Display results
    table = Table(box=box.ROUNDED)
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Name", style="white")

    for i, r in enumerate(results, 1):
        name = r["metadata"].get("name", "Unknown")
        table.add_row(str(i), f"{r['score']:.3f}", name)

    console.print(table)


@app.command()
def workspace(
    list: bool = typer.Option(False, "--list", "-l", help="List workspaces"),
    create: str | None = typer.Option(None, "--create", "-c", help="Create new workspace"),
    info: str | None = typer.Option(None, "--info", "-i", help="Show workspace info"),
) -> None:
    """Manage workspaces."""
    from platformdirs import user_data_dir

    base_dir = Path(user_data_dir("searchengine", "changqiangxia"))

    if list or (not create and not info):
        console.print("\n[bold]Workspaces:[/bold]\n")

        if not base_dir.exists():
            console.print("  [dim]No workspaces found[/dim]")
            return

        workspaces = [d for d in base_dir.iterdir() if d.is_dir()]

        if not workspaces:
            console.print("  [dim]No workspaces found[/dim]")
            return

        for ws in workspaces:
            index_exists = (ws / "indices" / "index.faiss").exists()
            status = "[green]✓[/green]" if index_exists else "[dim]empty[/dim]"
            console.print(f"  {status} {ws.name}")

    if create:
        ws_path = base_dir / create
        ws_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✅ Created workspace: {create}[/green]")

    if info:
        ws_path = base_dir / info
        if not ws_path.exists():
            console.print(f"[red]❌ Workspace not found: {info}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Workspace: {info}[/bold]\n")
        console.print(f"Path: {ws_path}")

        # Check for index
        index_path = ws_path / "indices" / "index.faiss"
        if index_path.exists():
            console.print("Index: [green]Exists[/green]")
            # Load and show stats
            try:
                engine = NexusEngine(workspace_dir=ws_path)
                if engine.index:
                    console.print(f"Vectors: {len(engine.index)}")
                    console.print(f"Type: {engine.index.index_type}")
            except Exception as e:
                console.print(f"[yellow]Could not load index: {e}[/yellow]")
        else:
            console.print("Index: [dim]Not built[/dim]")


@app.command()
def interpolate(
    concept_a: str = typer.Argument(..., help="Starting concept"),
    concept_b: str = typer.Argument(..., help="Ending concept"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    steps: int = typer.Option(5, "--steps", "-s", help="Number of interpolation steps"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Results per step"),
) -> None:
    """Interpolate between two concepts to discover intermediate ideas."""
    engine = get_engine(workspace)

    if engine.index is None:
        console.print("[red]❌ No index found. Run 'nexus index' first.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Interpolating: '{concept_a}' → '{concept_b}'[/bold]\n")

    try:
        path = engine.interpolate_concepts(concept_a, concept_b, steps=steps, top_k=top_k)

        for point in path:
            console.print(f"[cyan]{point['description']}[/cyan]")
            if point["neighbors"]:
                best = point["neighbors"][0]
                console.print(f"  Best match: {best['metadata']['name']} ({best['score']:.3f})")
            console.print()

    except Exception as e:
        console.print(f"[red]❌ Interpolation failed: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def blend(
    concepts: list[str] = typer.Argument(  # noqa: B008
        ..., help="Concepts with optional weights (e.g., 'vintage:0.8' 'neon:0.2')"
    ),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """Blend multiple concepts together with weights."""
    engine = get_engine(workspace)

    if engine.index is None:
        console.print("[red]❌ No index found. Run 'nexus index' first.[/red]")
        raise typer.Exit(1)

    # Parse concepts with weights
    parsed: list[tuple[str, float]] = []
    for concept in concepts:
        if ":" in concept:
            name, weight_str = concept.rsplit(":", 1)
            try:
                weight = float(weight_str)
            except ValueError:
                name = concept
                weight = 1.0
        else:
            name = concept
            weight = 1.0
        parsed.append((name, weight))

    # Normalize weights
    total = sum(w for _, w in parsed)
    parsed = [(c, w / total) for c, w in parsed]

    desc = " + ".join([f"{c}({w:.1%})" for c, w in parsed])
    console.print(f"\n[bold]Blending: {desc}[/bold]\n")

    try:
        results = engine.blend_concepts(parsed, top_k=top_k)

        table = Table(box=box.ROUNDED)
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Name", style="white")

        for i, r in enumerate(results, 1):
            name = r["metadata"].get("name", "Unknown")
            table.add_row(str(i), f"{r['score']:.3f}", name)

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Blend failed: {e}[/red]")
        raise typer.Exit(1) from None


@app.command()
def discover(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    method: str = typer.Option("kmeans", "--method", "-m", help="Clustering method"),
) -> None:
    """Discover semantic clusters in the index."""
    engine = get_engine(workspace)

    if engine.index is None:
        console.print("[red]❌ No index found. Run 'nexus index' first.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Discovering semantic clusters ({method})...[/bold]\n")

    try:
        clusters = engine.cluster_index(_method=method)

        if not clusters:
            console.print("[yellow]⚠️  Clustering requires embedding cache.[/yellow]")
            console.print("[dim]This feature is limited in the current version.[/dim]")
            return

        for cluster in clusters:
            console.print(f"[cyan]{cluster['label']}[/cyan]: {cluster['size']} items")

    except Exception as e:
        console.print(f"[red]❌ Discovery failed: {e}[/red]")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
