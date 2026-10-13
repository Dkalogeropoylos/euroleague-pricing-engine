"""Self-contained regression tests: do NOT depend on private uploaded CSV paths."""
import unittest
import pandas as pd
from audit_dataset import audit


def fixture():
    common={'SEASON':2026,'GAME_ID':'E2026_1','GAME_DATE':'2026-10-01T17:00:00+00:00','POSS_RAW':70.0,'TEAM_MINUTES':200.0,'OT_FLAG':False}
    basic={ 'PTS':70.,'FG2M':20.,'FG2A':35.,'FG3M':7.,'FG3A':20.,'FTM':9.,'FTA':12.,'OREB':8.,'DREB':20.,'REB':28.,'AST':15.,'STL':4.,'BLK':2.,'TO':12.,'PF':17.,'FD':17.}
    t=[]; p=[]
    for team,opp,side in [('A','B','H'),('B','A','A')]:
        row={**common,**basic,'TEAM_ID':team,'OPP_ID':opp,'HOME_AWAY':side}
        row.update({'OPP_'+stat:float(val) for stat,val in basic.items()})
        t.append(row)
        p.append({**common,**basic,'TEAM_ID':team,'OPP_ID':opp,'HOME_AWAY':side,'PLAYER_ID':'E:'+team+'01','MINUTES':200.,'POSITION':1})
    return pd.DataFrame(t),pd.DataFrame(p)

class AuditTests(unittest.TestCase):
    def test_valid_synthetic(self):
        a,b=fixture();r=audit(a,b)
        self.assertEqual(r['fatal'],[])
        self.assertFalse(r['ready_for_training'])
    def test_catches_score_corruption(self):
        a,b=fixture();a.loc[0,'PTS']+=1
        self.assertTrue(any('SCORING_IDENTITY' in x for x in audit(a,b)['fatal']))
    def test_catches_replicated_team(self):
        a,b=fixture();a=pd.concat([a,a.iloc[[0]]],ignore_index=True)
        self.assertIn('DUPLICATE_GAME_TEAM',audit(a,b)['fatal'])
    def test_catches_player_block_corruption(self):
        a,b=fixture();b.loc[0,'BLK']+=1
        self.assertTrue(any('PLAYER_TEAM_BLK_MISMATCH' in x for x in audit(a,b)['fatal']))
    def test_catches_minutes_corruption(self):
        a,b=fixture();b.loc[0,'MINUTES']+=5
        self.assertIn('TEAM_MINUTES_NOT_EQUAL_PLAYER_MINUTES',audit(a,b)['fatal'])

if __name__=='__main__':unittest.main()
