import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]



def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackDoubleCountingEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, num_decks=6, shuffle_on=15, natural=False):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Box(-10.0, +10.0, shape=(1,1), dtype=np.float32)
        ))
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        
        # Number of decks (classic 6 decks)
        self.num_decks = num_decks
        
        # When to shuffle all cards
        self.shuffle_on = shuffle_on
        
        # Store decks
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4 * self.num_decks
        
        # Store current count
        self.count = 0.0
        
        # Store information that player
        self.done = False
        
        # Start the first game
#         self.reset()
        
    def draw_card(self, np_random):
        # If deck is small, reset deck
        if len(self.deck) < self.shuffle_on:
            self.reset_deck()
            self.count = 0.0
            
            # Delete player left cards from deck and count
            for card in self.player:
                self.deck.pop(self.deck.index(card))
                self.count += self.halves[card]
            
            # Delete dealer cards
            for card in self.dealer:
                self.deck.pop(self.deck.index(card))
                
            # Count all previous dealer cards
            for card in self.dealer[:-1]:
                self.count += self.halves[card]
                
            # If player is done, we need to count dealer last card
            if self.done:
                self.count += self.halves[self.dealer[-1]]
        
        card = self.deck.pop(np.random.randint(0, len(self.deck)))
        return card
    
    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np.random)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            # Get player card
            player_card = self.draw_card(self.np_random)
            # Count it
            self.count += self.halves[player_card]
            # Memorize it
            self.player.append(player_card)
                               
            if is_bust(self.player):
                self.done = True
                reward = -1.
                
                # Add second dealer card to count
                self.count += self.halves[self.dealer[1]]
            else:
                self.done = False
                reward = 0.

        elif action == 0:  # stick: play out the dealers hand, and score
            self.done = True
            
            # Add second dealer card to count
            self.count += self.halves[self.dealer[1]]
            
            while sum_hand(self.dealer) < 17:
                # Get dealer card
                dealer_card = self.draw_card(self.np_random)
                # Count it
                self.count += self.halves[dealer_card]
                # Memorize it
                self.dealer.append(dealer_card)
                
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        else: # double
            # Get player card
            player_card = self.draw_card(self.np_random)
            # Count it
            self.count += self.halves[player_card]
            # Memorize it
            self.player.append(player_card)
            
            # Add second dealer card to count
            self.count += self.halves[self.dealer[1]]
            
            self.done = True
            if is_bust(self.player):
                reward = -2.0
                
                # Add second dealer card to count
                self.count += self.halves[self.dealer[1]]
            else:
                while sum_hand(self.dealer) < 17:
                    # Get dealer card
                    dealer_card = self.draw_card(self.np_random)
                    # Count it
                    self.count += self.halves[dealer_card]
                    # Memorize it
                    self.dealer.append(dealer_card)
                    
                reward = 2 * cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.count / math.ceil(len(self.deck) / 52))
#         return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), self.count)

    def reset(self):
        self.done = False
        
        if len(self.deck) < self.shuffle_on:
            self.reset_deck()
            self.count = 0.0
            
        # Draw dealer cards
        self.dealer = self.draw_hand(self.np_random)
        self.count += self.halves[self.dealer[0]]
        
        # Draw player cards
        self.player = self.draw_hand(self.np_random)
        self.count += self.halves[self.player[0]]
        self.count += self.halves[self.player[1]]
        
        return self._get_obs()
    
    def reset_deck(self):
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4 * self.num_decks
        
    @property
    def halves(self):
        return {
            1: -1,
            2: 0.5,
            3: 1,
            4: 1,
            5: 1.5,
            6: 1,
            7: 0.5,
            8: 0,
            9: -0.5,
            10: -1
        }