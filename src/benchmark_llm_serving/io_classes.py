import numpy as np
from dataclasses import dataclass, field, asdict

@dataclass
class QueryOutput:
    prompt: str = ""
    starting_timestamp: float = 0.0
    ending_timestamp: float = 0.0
    prompt_length: int = 0
    response_length: int = 0
    generated_text: str = ""
    total_query_time: float = 0.0
    timestamp_of_first_token: float = 0.0 
    time_to_first_token: float = 0.0
    timestamp_of_tokens_arrival: list = field(default_factory=list)
    delta_time_of_tokens_arrival: list = field(default_factory=list)
    completion_time_from_first_token: float = 0.0
    median_time_between_tokens: float = 0.0
    total_waiting_time: float = 0.0
    speed_from_beginning: float = 0.0
    speed_from_first_token: float = 0.0
    speed_without_waiting_time: float = 0.0
    success: bool = False
    error: str = ""
    timeout: bool = False

    def calculate_derived_stats(self):
        """Calculates characteristics from the initial ones.
        """
        self.response_length = len(self.timestamp_of_tokens_arrival)
        self.total_query_time = self.ending_timestamp - self.starting_timestamp
        self.delta_time_of_tokens_arrival = [self.timestamp_of_tokens_arrival[i+1] - self.timestamp_of_tokens_arrival[i] 
                                                for i in range(len(self.timestamp_of_tokens_arrival) - 1)]
        if len(self.timestamp_of_tokens_arrival):
            self.timestamp_of_first_token = self.timestamp_of_tokens_arrival[0]
            self.completion_time_from_first_token = self.ending_timestamp - self.timestamp_of_first_token
            self.time_to_first_token = self.timestamp_of_tokens_arrival[0] - self.starting_timestamp
        if len(self.delta_time_of_tokens_arrival):
            self.median_time_between_tokens = np.median(self.delta_time_of_tokens_arrival)
            self.total_waiting_time = self.calculate_total_waiting_time()
            self.speed_from_beginning = self.response_length / self.total_query_time
            self.speed_from_first_token = self.response_length / (self.total_query_time - self.time_to_first_token)
            self.speed_without_waiting_time = self.response_length / (self.total_query_time - self.total_waiting_time)

    def calculate_total_waiting_time(self):
        """Calculates the total waiting time which is the initial (ie the time to first token) plus all the times
        that the generation is paused
        """
        # We assume that the median is relevant only if we have at least 10 tokens generated
        if len(self.timestamp_of_tokens_arrival) > 10:
            std = np.std(self.delta_time_of_tokens_arrival, ddof=1)
            # We assume that there is waiting time if the generation time for a token is superior
            # to the median plus 3 standard deviation
            typical_generation_time = self.median_time_between_tokens + 3 * std
            waiting_time_except_first_token = sum([(delta - typical_generation_time) 
                                                   for delta in self.delta_time_of_tokens_arrival
                                                   if delta > typical_generation_time])
            return self.time_to_first_token + waiting_time_except_first_token
        else:
            return self.time_to_first_token

    def to_dict(self):
        """Returns the dataclass in a dictionary form
        """
        return asdict(self)


@dataclass
class QueryInput:
    prompt: str
    internal_id: int
    scheduled_delta: float = 0
    scheduled_timestamp: float = 0

    def add_starting_timestamp(self, starting_timestamp: float):
        """Adds the scheduled timestamp by adding the starting timestamp to the scheduled delta 
        """
        self.scheduled_timestamp = starting_timestamp + self.scheduled_delta
    
