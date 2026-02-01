"""Plugin System Base Classes.

Provides extensible architecture for custom processors, sources, and exporters.
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PluginInfo:
    """Plugin metadata.

    Attributes:
        name: Unique plugin name
        version: Plugin version
        description: Short description
        author: Plugin author
        dependencies: Required dependencies
    """

    name: str
    version: str
    description: str
    author: str = ""
    dependencies: list[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class Plugin(ABC):
    """Base class for all plugins.

    Plugins extend NexusMind functionality without modifying core code.
    """

    info: PluginInfo

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize plugin.

        Args:
            config: Plugin configuration
        """
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin.

        Returns:
            True if initialization successful
        """
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is unloaded."""
        pass

    def get_config_schema(self) -> dict[str, Any]:
        """Get configuration schema for validation."""
        return {}


class ImageSourcePlugin(Plugin):
    """Plugin for adding new image sources.

    Examples: Google Drive, S3, Dropbox, etc.
    """

    @abstractmethod
    def list_images(self, path: str) -> list[dict[str, Any]]:
        """List available images from source.

        Args:
            path: Source-specific path

        Returns:
            List of image metadata dicts
        """
        pass

    @abstractmethod
    def load_image(self, identifier: str) -> Any:
        """Load image from source.

        Args:
            identifier: Image identifier

        Returns:
            PIL Image or similar
        """
        pass

    def supports_streaming(self) -> bool:
        """Whether source supports streaming loads."""
        return False


class ProcessorPlugin(Plugin):
    """Plugin for processing images during indexing.

    Examples: OCR, face detection, NSFW filtering, etc.
    """

    @abstractmethod
    def process(self, image: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        """Process image and return enhanced metadata.

        Args:
            image: PIL Image
            metadata: Current metadata

        Returns:
            Enhanced metadata dict
        """
        pass

    def batch_process(
        self, images: list[Any], metadatas: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process multiple images (optional optimization)."""
        return [self.process(img, meta) for img, meta in zip(images, metadatas)]


class ExporterPlugin(Plugin):
    """Plugin for exporting search results.

    Examples: CSV, JSON, HTML gallery, etc.
    """

    @abstractmethod
    def export(self, results: list[dict[str, Any]], output_path: str) -> bool:
        """Export results to file.

        Args:
            results: Search results
            output_path: Output file path

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions."""
        pass


class SearchStrategyPlugin(Plugin):
    """Plugin for custom search strategies.

    Examples: Temporal search, geographic search, etc.
    """

    @abstractmethod
    def search(
        self,
        query: Any,
        index: Any,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Execute custom search.

        Args:
            query: Search query
            index: FAISS index
            **kwargs: Additional parameters

        Returns:
            Search results
        """
        pass


class PluginRegistry:
    """Registry for managing plugins.

    Singleton pattern for global plugin access.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins: dict[str, Plugin] = {}
            cls._instance._hooks: dict[str, list[Callable]] = {}
        return cls._instance

    def register(self, plugin: Plugin) -> bool:
        """Register a plugin.

        Args:
            plugin: Plugin instance

        Returns:
            True if successful
        """
        name = plugin.info.name

        if name in self._plugins:
            print(f"⚠️  Plugin {name} already registered")
            return False

        # Initialize
        if plugin.initialize():
            self._plugins[name] = plugin
            print(f"✅ Plugin registered: {name} v{plugin.info.version}")
            return True
        else:
            print(f"❌ Failed to initialize plugin: {name}")
            return False

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        if name in self._plugins:
            self._plugins[name].shutdown()
            del self._plugins[name]
            return True
        return False

    def get(self, name: str) -> Plugin | None:
        """Get plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[PluginInfo]:
        """List all registered plugins."""
        return [p.info for p in self._plugins.values()]

    def get_plugins_by_type(self, plugin_type: type) -> list[Plugin]:
        """Get plugins of specific type."""
        return [p for p in self._plugins.values() if isinstance(p, plugin_type)]

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add hook for event.

        Args:
            event: Event name
            callback: Function to call
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def execute_hooks(self, event: str, *args, **kwargs) -> list[Any]:
        """Execute all hooks for event."""
        results = []
        for hook in self._hooks.get(event, []):
            try:
                result = hook(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Hook error: {e}")
        return results


class PluginLoader:
    """Load plugins from directories."""

    def __init__(self, plugin_dirs: list[Path]):
        """Initialize loader.

        Args:
            plugin_dirs: Directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs
        self.registry = PluginRegistry()

    def load_all(self) -> int:
        """Load all plugins from directories.

        Returns:
            Number of plugins loaded
        """
        count = 0

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            for plugin_file in plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    self._load_plugin_file(plugin_file)
                    count += 1
                except Exception as e:
                    print(f"❌ Failed to load {plugin_file}: {e}")

        return count

    def _load_plugin_file(self, path: Path) -> None:
        """Load single plugin file."""
        # Dynamic import
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find plugin classes
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and hasattr(obj, "info")
            ):

                # Instantiate and register
                plugin = obj()
                self.registry.register(plugin)


# Global registry access
def get_plugin_registry() -> PluginRegistry:
    """Get global plugin registry."""
    return PluginRegistry()
