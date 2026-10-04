from enum import Enum, auto
import heapq

class EventType(Enum):
    """Enum class for different types of events in the FL system."""
    DOWNLOAD_JOB = auto()      # Client downloads the global model
    LOCAL_TRAIN = auto()       # Client trains locally
    UPLOAD_UPDATE = auto()     # Client uploads its model update
    AGGREGATE = auto()         # Server aggregates model updates
    SIMULATION_END = auto()    # End of simulation

class Event:
    """Class representing a scheduled event in the FL system."""
    
    def __init__(self, event_type, start_time, duration=None, client=None, data=None, config=None, event_id=None):
        """
        Initialize an event.
        
        Args:
            event_type (EventType): Type of the event
            start_time (float): Simulated time when the event starts
            duration (float): Duration of the event, None for automatic delay generation
            client_id (int, optional): ID of the client involved in the event
            data (any, optional): Additional data associated with the event
            event_id (int, optional): Unique identifier for the event
        """
        self.event_type = event_type
        self.start_time = start_time
        self.client = client
        self.event_id = event_id
        self.config = config
        if data is None:
            data = {}
        if duration is None:
            if event_type == EventType.DOWNLOAD_JOB:
                duration = self.client.download_delay_generator()
            elif event_type == EventType.LOCAL_TRAIN:
                duration = self.client.train_delay_generator()
            elif event_type == EventType.UPLOAD_UPDATE:
                duration = self.client.upload_delay_generator()
            else:
                raise ValueError(f"Invalid event type: {event_type} for None duration")
        self.duration = duration
        self.finish_time = start_time + self.duration
        self.data = data

    def __lt__(self, other):
        """Comparison operator for priority queue ordering based on finish time."""
        return self.finish_time < other.finish_time
    
    def __eq__(self, other):
        """Equality operator for priority queue."""
        if not isinstance(other, Event):
            return False
        return (self.finish_time == other.finish_time and 
                self.event_type == other.event_type and 
                self.client.client_id == other.client.client_id)
    
    def __str__(self):
        """String representation of the event."""
        client_id = self.client.client_id if self.client else None
        return f"Event(id={self.event_id}, type={self.event_type.name}, start={self.start_time:.2f}, duration={self.duration:.2f}, finish={self.finish_time:.2f}, client={client_id})"


class EventQueue:
    """Priority queue for scheduling and processing events in order of time."""
    
    def __init__(self):
        """Initialize an empty event queue."""
        self.queue = []
    
    def add_event(self, event):
        """
        Add an event to the queue.
        
        Args:
            event (Event): The event to add
        """
        heapq.heappush(self.queue, event)
    
    def get_next_event(self):
        """
        Get the next event from the queue.
        
        Returns:
            Event: The next event in order of time
        """
        if not self.is_empty():
            return heapq.heappop(self.queue)
        return None
    
    def is_empty(self):
        """
        Check if the queue is empty.
        
        Returns:
            bool: True if the queue is empty, False otherwise
        """
        return len(self.queue) == 0
    
    def peek(self):
        """
        Peek at the next event without removing it.
        
        Returns:
            Event: The next event in order of time
        """
        if not self.is_empty():
            return self.queue[0]
        return None
    
    def current_time(self):
        """
        Get the current simulation time based on the next event.
        
        Returns:
            float: The finish time of the next event, or 0 if the queue is empty
        """
        if not self.is_empty():
            return self.queue[0].finish_time
        return 0 