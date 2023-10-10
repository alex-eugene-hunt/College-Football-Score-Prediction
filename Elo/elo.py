# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:19:01 2023

@author: Eric
"""

import math

class Elo:
    
    def __init__(self, k=20):
        self.k = k
        self.players = {}
     
    def add_player(self, name, elo=1500):
        """
        Adds player to current Elo system
        :param name: name of the player
        :param elo: starting elo rating
        """
        # Only add if player is not already in the system
        if name in self.players:
            print("Player is already in the system")
            return
        
        self.players[name] = elo
          
    def reset_player(self, name, elo=1500):
        """
        Resets player's rating
        :param name: name of the player
        :param elo: elo rating to reset to
        """
        # Only reset if player is already in the system
        if name not in self.players:
            print("Player is not in the system")
            return
        
        self.players[name] = elo
    
    def remove_player(self, name):
        """
        Removes player from the Elo system
        :param name: name of the player
        """
        # Only remove if player is already in the system
        if name not in self.players:
            print("Player is not in the system")
            return
        
        self.players.pop(name)    
    
    def get_players(self):
        """
        Returns the entire dictionary of players and their Elo ratings
        :return: dictionary of players and corresponding Elo ratings
        """
        return self.players
        
    def expect(self, player1, player2):
        """
        Makes prediction for win chance of player 1 vs. player 2
        :param player1: player to evaluate from
        :param player2: player1's opponent
        :return: percentage chance of player 1 winning against player 2
        """
        exp = (self.players[player2] - self.players[player1]) / 400
        return 1/(1 + 10**(exp))
    
    def rate(self, player1, player2, outcome):
        """
        Records a game in the Elo system and changes ratings accordingly
        :param player1: player to evaluate from
        :param player2: player1's opponent
        :param outcome: whether player 1 won against player 2
        """
        if player1 not in self.players and player2 not in self.players:
            print("Players not found")
            return
        
        r1 = 10**(self.players[player1]/400)
        r2 = 10**(self.players[player2]/400)
        
        e1 = r1/(r1+r2)
        e2 = r2/(r1+r2)
        
        score_1 = outcome
        score_2 = 1 - outcome
            
        self.players[player1] += self.k * (score_1 - e1)
        self.players[player2] += self.k * (score_2 - e2)
        