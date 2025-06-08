from abc import ABC, abstractmethod

from ..pyexecutor.sampler import SampleState
from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):

    def __init__():
        pass

    @abstractmethod
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        state: SampleState,
    ) -> None:
        """
        Prepare the drafter tokens for the forward computation this step.
        """
        raise NotImplementedError
