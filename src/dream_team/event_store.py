"""
Event store for Dream Team framework.

Implements append-only event logging as ground truth for all system activities.
No truncation ever - all data is preserved.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
import uuid


@dataclass
class Event:
    """
    Single event in the experiment timeline.
    
    All activities are logged: execution, meetings, reflections, evolution, KB updates.
    Large data (outputs, logs) stored as separate files; paths stored in payload.
    """
    id: str
    experiment_id: str
    iteration: int
    kind: str  # "execution", "meeting", "reflection", "prompt", "evolution", "kb_update"
    agent: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Load event from dictionary."""
        return cls(**data)


class EventStore:
    """
    Append-only event storage.
    
    Ground truth for all experiment activities. No truncation, ever.
    If output is huge, write to file and store path in payload.
    """
    
    def __init__(self, experiment_id: str, storage_dir: Optional[Path] = None):
        """
        Initialize event store.
        
        Args:
            experiment_id: Unique identifier for this experiment
            storage_dir: Directory to store event files (default: ./events)
        """
        self.experiment_id = experiment_id
        self.storage_dir = storage_dir or Path("./events")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.events: List[Event] = []
        
        # Create directory for large payloads
        self.blob_dir = self.storage_dir / experiment_id / "blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
    
    def log_event(
        self,
        kind: str,
        iteration: int = 0,
        agent: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        large_data: Optional[Dict[str, str]] = None
    ) -> Event:
        """
        Log a new event.
        
        Args:
            kind: Event type ("execution", "meeting", "reflection", etc.)
            iteration: Iteration number (0 for bootstrap/pre-experiment)
            agent: Agent name if event is agent-specific
            payload: Event data (dictionaries only, no objects)
            large_data: Large text data to store separately
                Format: {"key": "large_text_content"}
                These will be written to files and paths stored in payload
        
        Returns:
            Created event
        """
        event_id = str(uuid.uuid4())
        
        # Handle large data by storing to files
        final_payload = payload or {}
        if large_data:
            for key, content in large_data.items():
                blob_path = self._store_blob(event_id, key, content)
                final_payload[f"{key}_path"] = str(blob_path)
                # Also store truncated preview
                preview = content[:500] + "..." if len(content) > 500 else content
                final_payload[f"{key}_preview"] = preview
        
        event = Event(
            id=event_id,
            experiment_id=self.experiment_id,
            iteration=iteration,
            kind=kind,
            agent=agent,
            payload=final_payload
        )
        
        self.events.append(event)
        self.save()
        return event
    
    def _store_blob(self, event_id: str, key: str, content: str) -> Path:
        """
        Store large text content to file.
        
        Args:
            event_id: Event identifier
            key: Data key name
            content: Text content to store
        
        Returns:
            Path to stored file
        """
        filename = f"{event_id}_{key}.txt"
        blob_path = self.blob_dir / filename
        blob_path.write_text(content, encoding="utf-8")
        return blob_path
    
    def get_events(
        self,
        kind: Optional[str] = None,
        iteration: Optional[int] = None,
        agent: Optional[str] = None
    ) -> List[Event]:
        """
        Query events by filters.
        
        Args:
            kind: Filter by event type
            iteration: Filter by iteration number
            agent: Filter by agent name
        
        Returns:
            List of matching events
        """
        results = self.events
        
        if kind is not None:
            results = [e for e in results if e.kind == kind]
        
        if iteration is not None:
            results = [e for e in results if e.iteration == iteration]
        
        if agent is not None:
            results = [e for e in results if e.agent == agent]
        
        return results
    
    def get_blob_content(self, event: Event, key: str) -> Optional[str]:
        """
        Load large data content from blob file.
        
        Args:
            event: Event containing the blob reference
            key: Data key name
        
        Returns:
            Full content or None if no blob exists
        """
        path_key = f"{key}_path"
        if path_key not in event.payload:
            return None
        
        blob_path = Path(event.payload[path_key])
        if not blob_path.exists():
            return None
        
        return blob_path.read_text(encoding="utf-8")
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Persist event store to JSON.
        
        Args:
            filepath: Custom save path (default: storage_dir/experiment_id/events.json)
        
        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = self.storage_dir / self.experiment_id / "events.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment_id": self.experiment_id,
            "events": [e.to_dict() for e in self.events]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path, storage_dir: Optional[Path] = None) -> "EventStore":
        """
        Load event store from JSON.
        
        Args:
            filepath: Path to events.json file
            storage_dir: Directory for event storage (default: ./events)
        
        Returns:
            Loaded EventStore instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        store = cls(
            experiment_id=data["experiment_id"],
            storage_dir=storage_dir
        )
        
        store.events = [Event.from_dict(e) for e in data["events"]]
        return store
    
    def __len__(self) -> int:
        """Return number of events stored."""
        return len(self.events)
