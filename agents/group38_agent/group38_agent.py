from decimal import Decimal
import logging
import numpy as np
from random import randint
from time import time
from typing import Tuple, cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


class Group38Agent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        # Concession factor: beta
        self._beta = 0.2
        self.current_bids = []
        self.pareto_frontier = []
        self.kalai_smorodinsky = None


    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    # Do not change this function - capabilities have been set for the competition
    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Initial Agent - CSE3210 - Group 38"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """

        """An (Action) can either be an Offer, Accept or EndNegotiation (walkaway)"""

        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################



    """[Strategy]
    Determines when a bid is accepted and not
    """
    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # Accept this bid once its utility has reached at least our utility-goal.
        # This utility goal is based on the progress in the negotiation.
        conditions = [
            self.profile.getUtility(bid) > self._getUtilityGoal(progress),
            progress >= 0.75
        ]
        return all(conditions)

    """
    Find a new bid, by taking 500 attempts and choosing the bid with the highest score,
    according to 'score_bid'
    """
    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # Calculate our utility goal
        utilityGoal = self._getUtilityGoal(progress)

        best_bid_score = 0.0
        best_bid = None
        best_dist_to_utilityGoal = 0.4
        dist_to_utilityGoal = 1.0

        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            bid_score = self.score_bid(bid, eps=0.00001)
            dist_to_utilityGoal = abs(bid_score - float(utilityGoal))
            if (bid_score < utilityGoal and bid_score > best_bid_score) \
                or (bid_score >= utilityGoal and dist_to_utilityGoal < best_dist_to_utilityGoal):
                best_bid_score, best_bid, best_dist_to_utilityGoal = bid_score, bid, dist_to_utilityGoal

        return best_bid

    """[Strategy]
    Score a bid, based on both our own utility and the predicted utility of the opponent.
    """
    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.01) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)
        
        our_utility = float(self.profile.getUtility(bid))
        
        time_pressure = 1.0 - progress ** (1 / eps)
        old_score = alpha * time_pressure * our_utility
        
        if self.opponent_model is None:
            return old_score 

        opponent_utility = self.opponent_model.get_predicted_utility(bid)
        opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
        old_score += opponent_score

        # Add current bid to list
        self.current_bids.append((our_utility, opponent_utility))


        new_kalai = None
        pareto_distance = np.sqrt(2.0)
        max_pareto = 0.0        
        closest_up = (0.0,1.0) 
        closest_down = (1.0,0.0)
        pareto_changed = False
        if len(self.pareto_frontier) == 0:
            # first bid so add to POF
            pareto_changed = True
            self.pareto_frontier.append((our_utility,opponent_utility))
            max_pareto = 1.0
            pareto_distance = 0.0
            new_kalai = ((our_utility+opponent_utility)/2.0,(our_utility+opponent_utility)/2.0)
        else:
            new_pareto = []
            include = True
            # loop through POF and check if new bid will change the frontier
            # while in this loop, we also check which points are closest to the y=x line, to later calculate the kalai smordin-whatever point
            for point in self.pareto_frontier:

                if our_utility <= point[0] and opponent_utility <= point[1]:
                    include = False
                if our_utility > point[0] and opponent_utility > point[1]:
                    # don't include this point in the POF anymore
                    continue   
                # update pareto distances
                pareto_distance = min(pareto_distance,np.sqrt(np.abs(point[0]-our_utility)**2 + np.abs(point[1]-opponent_utility)**2))
                max_pareto = max(max_pareto, np.sqrt(point[0]**2 + point[1]**2))

                # include in new POF
                new_pareto.append(point)

                if not include:
                    continue
                # check of points close to y=x
                if point[0] == point[1]:
                    new_kalai = point 
                    # break
                elif point[0] < point[1]:
                    closest_up = min(closest_up, point, key=lambda x: np.abs(x[0]-x[1]))
                else:
                    closest_down = min(closest_down, point, key=lambda x: np.abs(x[0]-x[1]))



            if include or len(self.pareto_frontier) != len(new_pareto):
                pareto_changed = True
            if include:
                # include new bid in POF and again check for this new bid if its close to y=x
                if our_utility == opponent_utility:
                    new_kalai = (our_utility,opponent_utility)
                elif our_utility < opponent_utility:
                    closest_up = min(closest_up, (our_utility,opponent_utility), key=lambda x: np.abs(x[0]-x[1]))
                else:
                    closest_down = min(closest_down, (our_utility,opponent_utility), key=lambda x: np.abs(x[0]-x[1]))

                max_pareto = 1.0
                pareto_distance = 0.0
                new_pareto.append((our_utility,opponent_utility))

                # store the new POF
                self.pareto_frontier = new_pareto

        # from the points found close to y=x, calculate kalai
        if new_kalai is None and pareto_changed:
            if closest_up != (0.0,1.0):
                if closest_down != (1.0,0.0):
                    # intersection point of y=x and line through two closest points to this line (on both sides)
                    if (closest_up[0]-closest_down[0]) == 0:
                        self.kalai_smorodinsky = (closest_up[0],closest_up[0])
                    else:
                        a = (closest_up[1]-closest_down[1])/(closest_up[0]-closest_down[0])
                        b = closest_up[1]-a*closest_up[0]
                        self.kalai_smorodinsky = (b/(1.0-a),b/(1.0-a))
                else:
                    self.kalai_smorodinsky = ((closest_up[0]+closest_up[1])/2.0,(closest_up[0]+closest_up[1])/2.0)
            elif closest_down != (1.0,0.0):
                self.kalai_smorodinsky = ((closest_down[0]+closest_down[1])/2.0,(closest_down[0]+closest_down[1])/2.0)
        elif pareto_changed:
            self.kalai_smorodinsky = new_kalai


        # # Calculate the Nash product
        # nash_products = np.array([outcome[0] * outcome[1] for outcome in pareto_frontier])

        # kalai_smorodinsky = np.array(kalai_smorodinsky)
        # # Calculate the distances from the bid to these features
        # pareto_distance = np.min(np.sqrt(np.sum((pof - np.array([our_utility, opponent_utility])) ** 2, axis=1)))
        # nash_distance = np.min(np.abs(nash_products - (our_utility * opponent_utility)))
        # ks_distance = np.sqrt(np.sum((kalai_smorodinsky - np.array([our_utility, opponent_utility])) ** 2))

        # # Define weights dependent on how important each factor is
        # pareto_weight = 0.5
        # nash_weight = 0.3
        # ks_weight = 0.2

        # # TODO: find proper max distance
        # n = AllBidsList(self.profile.getDomain()).size()
        # max_distance = np.sqrt(np.sum(n ** 2))

        # # Calculate seperate scores
        # pareto_score = (max_distance - pareto_distance) / max_distance
        # nash_score = (max_distance - nash_distance) / max_distance
        # ks_score = ks_distance / max_distance

        # Define a function that can be used to mix scores
        def mix_score(*scores: Tuple[float, float]) -> float:
            # Add all weights
            total_weight: float = 0.0
            for (wght, _) in scores:
                total_weight += wght

            # Compute the mix
            total_score: float = 0.0
            for (wght, sc) in scores:
                total_score += (wght/total_weight) * sc

            return total_score

        ks_distance = np.sqrt((our_utility-self.kalai_smorodinsky[0])**2 + (opponent_utility-self.kalai_smorodinsky[1])**2)
        ks_score = np.sqrt(self.kalai_smorodinsky[0]**2 + self.kalai_smorodinsky[1]**2) - ks_distance
        pareto_score = max_pareto - pareto_distance

        if progress < 0.5:
            return old_score

        return mix_score(
            (1, old_score),
            (1, ks_score),
            (0.5, pareto_score)
        )
        # return score
    #
    # PRIVATE FUNCTIONS
    #

    """ [Strategy]
    Calculate the utility-goal, based on the strategy of choice.
    """
    def _getUtilityGoal(self, progress: float) -> Decimal:

        # Max- and min-utilities
        minUtil = Decimal(0) # default reservation-value of 0.5
        maxUtil = Decimal(1)
        
        # Definition of the progress-vector
        def progressVector(progress):
            if self._beta == 0:
                return Decimal(1)
            else:
                return Decimal(round( 1.0 - pow(progress, 1.0/self._beta), 6))
        
        # Compute the maximum of the minimum of the max-utility and the
        # calculated utilility with concessions
        result = max(
            min(
                (minUtil + (maxUtil - minUtil) * progressVector(progress)),
                maxUtil
            ),
            minUtil
        )
        
        return result


        
