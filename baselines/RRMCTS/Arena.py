import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any single agent can be pit against itself.
    """
    def __init__(self, player, game, display=None):
        """

        """
        self.game = game
        self.display = display
        self.player=player

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            scores
        """
        state = self.game.getInitState(maxmum_items_number=self.game.maxmum_items_number)
        it = 0
        player=self.player
        while self.game.getGameEnded(state) == 0:
            it += 1
            action = player(state)
            state = self.game.getNextState(state, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(state)))
        result = self.game.getGameEnded(state)
        return result

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        scores = []
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            scores.append(gameResult)
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()

        return scores
