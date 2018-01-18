"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """
    This scoring function is adds a distance bias on top of the
    improved_score function. This will help us pick moves that
    would have the biggest difference in legal moves while
    favoring those that keep distance away from opponent.
    """
    score = game.utility(player)
    if score != 0:
        return score

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    diff_moves = float(own_moves - opp_moves)

    player_pos = game.get_player_location(player) or (0, 0)
    opponent_pos = game.get_player_location(game.get_opponent(player)) or (0, 0)
    distance = ((player_pos[0] - opponent_pos[0]) ** 2 + (player_pos[1] - opponent_pos[1]) ** 2) ** (1/2.0)
    d_max = (game.height ** 2 + game.width ** 2) ** (1/2.)
    distance_bias = distance / 2

    return diff_moves + distance_bias

def custom_score_2(game, player):
    """
    This scoring function is adds a center bias on top of the
    improved_score function. This will help us pick moves that
    are closer to the center.
    """
    score = game.utility(player)
    if score != 0:
        return score

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    diff_moves = float(own_moves - opp_moves)

    h, w = game.height / 2., game.width / 2.
    y, x = game.get_player_location(player)
    is_center = h - 1 <= y <= h + 1 and w - 1 <= x <= w + 1
    center_bias = 1.5 if is_center else 1

    return diff_moves * center_bias


def custom_score_3(game, player):
    """
    This scoring function calculates the difference between self and
    oppenent and try to maximize the distance.
    """
    score = game.utility(player)
    if score != 0:
        return score

    player_pos = game.get_player_location(player) or (0, 0)
    opponent_pos = game.get_player_location(game.get_opponent(player)) or (0, 0)
    distance = ((player_pos[0] - opponent_pos[0]) ** 2 + (player_pos[1] - opponent_pos[1]) ** 2) ** (1/2.)
    return distance


class IsolationPlayer:
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        # This function calls the minimax to get the best move based on
        # a predefined search depth.
        self.time_left = time_left
        best_move = (-1, -1)

        try:
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass

        return best_move

    def max_value(self, game, depth):
        # This function implements the behavior in the maximizing layer.
        # It loops over all legal moves, calls min_value on each, and
        # returns the highest score. Every recursion deducts one from
        # depth in order to reach the lowest possible layer. Once it
        # reaches the base case, it calls score function to get the
        # score and returns it.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if depth == 0 or not legal_moves:
            return self.score(game, self)
        return max([self.min_value(game.forecast_move(move), depth - 1) for move in legal_moves])

    def min_value(self, game, depth):
        # This is similar to max_value, except it returns the lowest
        # score after evaluating all the legal moves. This function
        # implements the details of the minmizing layer.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if depth == 0 or not legal_moves:
            return self.score(game, self)
        return min([self.max_value(game.forecast_move(move), depth - 1) for move in legal_moves])


    def minimax(self, game, depth):
        # This function is called at the root node because we need to
        # know which node to pick. The difference between this function
        # and the max_value is that this function returns the move with
        # the highest score instead of the actual score.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)

        _, move = max([(self.min_value(game.forecast_move(m), depth - 1), m) for m in legal_moves])
        return move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        # This function calls alphabeta function to return the best
        # moves. It also does iterative deepening to try to reach
        # the deepest level possible before timeout.
        self.time_left = time_left

        best_move = (-1, -1)
        freq_hash = {}
        highest_freq = 1

        try:
            while True:
                move = self.alphabeta(game, self.search_depth)
                if move != (-1, -1):
                    if move not in freq_hash:
                        freq_hash[move] = 0
                    freq_hash[move] += 1
                    if freq_hash[move] >= highest_freq:
                        best_move, highest_freq = move, freq_hash[move]
                self.search_depth += 1
        except SearchTimeout:
            pass
        return best_move

    def max_value(self, game, depth, alpha=("-inf"), beta=float("inf")):
        # This function is similar to MinimaxPlayer#max_value, except
        # it has included the alphabeta pruning logic. It calls
        # min_value with the highest alpha and lowest beta up to this
        # point. In the maximizing layer, beta is static and is recieved
        # from the layer above. For alpha, it is first received from
        # the layer above and this function calls min_value to get
        # a new alpha value named local_alpha. If local_alpha is higher
        # than alpha, we choose the local_alpha as the new alpha. Once
        # we know that alpha is larger than or equal to beta, we can
        # abort the search and return the alpha. This is because the
        # layer above is a minimizing layer and it already has a lower
        # beta (which is being passed down to this layer as beta) and
        # it will never pick a score that is higher than beta.

        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if depth == 0 or not legal_moves:
            return self.score(game, self)
        for m in legal_moves:
            if alpha >= beta: break
            local_alpha = self.min_value(game.forecast_move(m), depth - 1, alpha, beta)
            if local_alpha > alpha: alpha = local_alpha
            if self.time_left() < self.TIMER_THRESHOLD * 3: return alpha

        return alpha


    def min_value(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        # This fucntion is similar to the max_value, except it chooses
        # the lowest beta and return it to the maximizing layer above
        # as the local_alpha. It also implements AB pruning when beta
        # is less than or equal to alpha. This is because the layer
        # above will not consider this branch if it sees a local_alpha
        # less than or equal to the alpha it already has.
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if depth == 0 or not legal_moves:
            return self.score(game, self)
        for m in legal_moves:
            if beta <= alpha: break
            local_beta = self.max_value(game.forecast_move(m), depth - 1, alpha, beta)
            if local_beta < beta: beta = local_beta
            if self.time_left() < self.TIMER_THRESHOLD * 3: return beta

        return beta

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        # This function is similar to max_value except it returns the
        # best move instead of the highest score. It also doesn't
        # implement AB pruning since it is at the root node. It doesn't
        # have any information regarding beta since it doesn't have a
        # minimizing layer above to pass in the value. As such, it has
        # to perform search on all legal moves.
        if self.time_left() < self.TIMER_THRESHOLD: raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        best_move = (-1, -1)

        for m in legal_moves:
            local_alpha = self.min_value(game.forecast_move(m), depth - 1, alpha, beta)
            if local_alpha > alpha: alpha, best_move = local_alpha, m
            if self.time_left() < self.TIMER_THRESHOLD * 3: return best_move

        return best_move
