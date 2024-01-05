import matplotlib.pyplot as plt


class WordleDynamic(object):
    def __init__(self) -> None:
        """
        Dynamic:
        twitters = twitter_rate x player
        没玩过的用户: playing_t1 = playing x (1 - leaving_rate) + twitters x new_player_rate x (new_users / all_users)
        正在玩的用户: new_users_t1 = new_users - new_playing + refresh_rate x played x twitter
        玩腻了的用户: played_t1 = played + new_played - new_refresh
        """
        # need to be optimized
        self.all_users = 15000000
        self.twitter_rate = 0.25
        self.twitter_spread_rate = 3.2
        self.new_player_rate = 0.2
        self.old_player_rate = 0.1
        self.leaving_rate = 0.1
        # status
        self._player = 0
        self._played = 0
        self._new_users = self.all_users
        self._twitters = 0
        self._played_and_xinxian = 0

    def initialize(self, player:int) -> None:
        self._player = player
        self._played = 0
        self._new_users = self.all_users - player
        return None

    def update_status(self) -> None:
        current_tired = self._player * self.leaving_rate
        current_new_player = (
            min(self._twitters * self.twitter_spread_rate, self.all_users)
            * (self._new_users / self.all_users)
            * self.new_player_rate
        )
        self._new_users -= current_new_player
        current_refreshed = (
            min(self._twitters * self.twitter_spread_rate, self.all_users)
            * (self._played / self.all_users)
            * self.old_player_rate
        )
        self._new_users += current_refreshed
        self._player += current_new_player - current_tired
        self._played += current_tired
        self._played -= current_refreshed
        self._twitters = self._player * self.twitter_rate
        return None

    @property
    def twitters(self) -> None:
        return self._twitters

wordle = WordleDynamic()
wordle.initialize(10)
tws = []
for i in range(560):
    wordle.update_status()
    tws.append(wordle.twitters)
    print(f"{i} - {wordle.twitters} - {wordle._played}")
tws = tws[200:]
plt.plot(range(len(tws)), tws, "-")
plt.show()