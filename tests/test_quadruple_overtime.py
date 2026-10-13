from providers.euroleague import EuroLeagueProvider, valid_team_minutes


def test_overtime_team_minutes_valid():
    for ot in range(0, 5):
        assert valid_team_minutes(200 + 25 * ot)
    for wrong in (0, 150, 210, 248.8, 281, 301.2):
        assert not valid_team_minutes(float(wrong))


def test_game_with_four_overtimes_is_accepted():
    game = {"gameCode":170, "date":"2024-01-05T20:00:00Z", "played":True,
            "phaseType":"RS", "local":{"club":{"code":"MAD","name":"Real Madrid"},"score":130},
            "road":{"club":{"code":"IST","name":"Anadolu Efes"},"score":126}}
    def side(code, score):
        # Synthetic fixture validates MINUTES logic only, not a reconstructed match boxscore.
        n2 = 20 if score == 130 else 18
        n3 = 20 if score == 130 else 20
        ft = score - 2*n2 - 3*n3
        total = {"points":score,"fieldGoalsMade2":n2,"fieldGoalsAttempted2":55,
                 "fieldGoalsMade3":n3,"fieldGoalsAttempted3":40,
                 "freeThrowsMade":ft,"freeThrowsAttempted":ft+2,
                 "offensiveRebounds":20,"defensiveRebounds":30,
                 "blocksFavour":2,"turnovers":14}
        players = [{"player":{"person":{"code":f"{code}{i}","name":str(i)}},
                    "stats":{"timePlayed":3600,"points":0,"fieldGoalsMade2":0,
                             "fieldGoalsAttempted2":0,"fieldGoalsMade3":0,
                             "fieldGoalsAttempted3":0,"freeThrowsMade":0,"freeThrowsAttempted":0,
                             "offensiveRebounds":0,"defensiveRebounds":0,"blocksFavour":0,"turnovers":0}}
                   for i in range(5)]
        return {"total":total,"players":players}
    teams, players, errors = EuroLeagueProvider.normalize_game(game,{"local":side("MAD",130),"road":side("IST",126)},"E",2023)
    assert len(teams)==2 and len(players)==10
    assert [int(t["TEAM_MINUTES"]) for t in teams]==[300,300]
    assert all(t["OT_FLAG"] for t in teams)
    assert not any("minutes sum atypical" in e for e in errors)
