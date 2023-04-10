from collections import defaultdict
from typing import List

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValueSet import DiscreteValueSet
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value


class OpponentModel:
    def __init__(self, domain: Domain):
        self.offers = []
        self.domain = domain

        self.issue_estimators = {
            i: IssueEstimator(v) for i, v in domain.getIssuesValues().items()
        }

    def update(self, bid: Bid, progress: float):
        # keep track of all bids received
        self.offers.append(bid)

        # update all issue estimators with the value that is offered for that issue
        for issue_id, issue_estimator in self.issue_estimators.items():
            issue_estimator.update(bid.getValue(issue_id), progress)

    def get_predicted_utility(self, bid: Bid):
        if len(self.offers) == 0 or bid is None:
            return 0

        # initiate
        total_issue_weight = 0.0
        value_utilities = []
        issue_weights = []

        for issue_id, issue_estimator in self.issue_estimators.items():
            # get the value that is set for this issue in the bid
            value: Value = bid.getValue(issue_id)

            # collect both the predicted weight for the issue and
            # predicted utility of the value within this issue
            value_utilities.append(issue_estimator.get_value_utility(value))
            issue_weights.append(issue_estimator.weight)

            total_issue_weight += issue_estimator.weight

        # normalise the issue weights such that the sum is 1.0
        if total_issue_weight == 0.0:
            issue_weights = [1 / len(issue_weights) for _ in issue_weights]
        else:
            issue_weights = [iw / total_issue_weight for iw in issue_weights]

        # calculate predicted utility by multiplying all value utilities with their issue weight
        predicted_utility = sum(
            [iw * vu for iw, vu in zip(issue_weights, value_utilities)]
        )

        return predicted_utility


class IssueEstimatorWindow:
    def __init__(self, value_set: DiscreteValueSet):
        self.bids_received = 0
        self.max_value_count = 0
        self.num_values = value_set.size()
        self.value_trackers = defaultdict(ValueEstimator)

        self._k = 10
        
    def update(self, value: Value) -> bool:
        self.bids_received += 1

        # get the value tracker of the value that is offered
        value_tracker: ValueEstimator = self.value_trackers[value]

        # register that this value was offered
        value_tracker.update()

        # update the count of the most common offered value
        self.max_value_count = max([value_tracker.count, self.max_value_count])

        # If this window is full, meaning it contains k bids, return True
        # signaling that a new window should be opened
        return self.bids_received == self._k
    
    # Function: determine whether the difference in max_value_count
    # between two windows is within the specified bounds
    def compare_within_bounds(self, other: 'IssueEstimatorWindow', max_dist: int) -> bool:
        return abs(self.max_value_count - other.max_value_count) <= max_dist


class IssueEstimator:
    def __init__(self, value_set: DiscreteValueSet):
        if not isinstance(value_set, DiscreteValueSet):
            raise TypeError(
                "This issue estimator only supports issues with discrete values"
            )

        self.bids_received = 0
        self.max_value_count = 0
        self.num_values = value_set.size()
        self.value_trackers = defaultdict(ValueEstimator)
        self.weight = 0

        self.value_set = value_set
        self.windows: List[IssueEstimatorWindow] = []

        self.update_params = {"alpha":0.3, "beta":0.1}

    def update(self, value: Value, progress: float):
        self.bids_received += 1
        
        # If this is the first time update is called, initialize a new window
        if len(self.windows) == 0:
            self.windows.append(IssueEstimatorWindow(self.value_set))

        # Update the last window
        # If this update-call returns True, this window is full and a new one should be created
        if self.windows[len(self.windows)-1].update(value):
            self.windows.append(IssueEstimatorWindow(self.value_set))

        # Also still do all these steps in a non-window context
        # get the value tracker of the value that is offered
        value_tracker: ValueEstimator = self.value_trackers[value]

        # register that this value was offered
        value_tracker.update()

        # update the count of the most common offered value
        self.max_value_count = max([value_tracker.count, self.max_value_count])

        # update predicted issue weight
        # the intuition here is that if the values of the receiverd offers spread out over all
        # possible values, then this issue is likely not important to the opponent (weight == 0.0).
        # If all received offers proposed the same value for this issue,
        # then the predicted issue weight == 1.0
        # equal_shares = self.bids_received / self.num_values
        # self.weight = (self.max_value_count - equal_shares) / (
        #     self.bids_received - equal_shares
        # )



        # CALCULATING THE NEW WEIGHT, ACCORDING TO  [Tunalı et al., 2017]

        # So long as the second window is still open yet, there's nothing to compare so we use the old
        # way of computing weights
        if len(self.windows) <= 2:
            equal_shares = self.bids_received / self.num_values
            self.weight = (self.max_value_count - equal_shares) / (
                self.bids_received - equal_shares
            )

        else:
            # Select the last two windows that are closed
            # self.windows[-1] is the window that is currently open
            # self.windows[-2] is the last closed window
            # self.windows[-3] is the second-to-last closed window
            window1 = self.windows[-3]
            window2 = self.windows[-2]

            # If window2 has approximately the same max_value_count as window1
            # (max distance 2), then we consider the behaviour of our opponent
            # unchanged from window1 to window2
            if window1.compare_within_bounds(window2, 2):
                self.weight = max(1.0,
                                  self.weight + (self.update_params["alpha"] * (1.0 - progress ** self.update_params["beta"]))
                                  )
                
            # If it seems like the opponent changed its behaviour and we conclude this issue is not important to our opponent.
            else:
                self.weight = 0.0

        # recalculate all value utilities
        for value_tracker in self.value_trackers.values():
            value_tracker.recalculate_utility(self.max_value_count, self.weight)

    def get_value_utility(self, value: Value):
        if value in self.value_trackers:
            return self.value_trackers[value].utility

        return 0


class ValueEstimator:
    def __init__(self):
        self.count = 0
        self.utility = 0

    def update(self):
        self.count += 1

    def recalculate_utility(self, max_value_count: int, weight: float):
        if weight < 1:
            mod_value_count = ((self.count + 1) ** (1 - weight)) - 1
            mod_max_value_count = ((max_value_count + 1) ** (1 - weight)) - 1

            self.utility = mod_value_count / mod_max_value_count
        else:
            self.utility = 1
