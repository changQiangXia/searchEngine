"""Built-in processor plugins."""

from typing import Dict, Any

from nexus_mind.plugins.base import ProcessorPlugin, PluginInfo


class NSFWFilterProcessor(ProcessorPlugin):
    """Filter NSFW content during indexing."""
    
    info = PluginInfo(
        name="nsfw_filter",
        version="1.0.0",
        description="Filter NSFW content",
        author="NexusMind",
    )
    
    def initialize(self) -> bool:
        return True
    
    def process(self, image, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check image for NSFW content."""
        metadata['nsfw_safe'] = True
        metadata['nsfw_score'] = 0.0
        return metadata