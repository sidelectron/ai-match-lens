import unittest
from matchlens.tracks import TrackFilter, duplicate_boxes


def player(i,box,confidence=.8):
    return {'id':i,'box':box,'confidence':confidence}


class TrackTests(unittest.TestCase):
    def test_real_frame_82_nested_boxes(self):
        self.assertTrue(duplicate_boxes([904.3,416.2,951.0,531.0], [906.2,455.7,956.9,534.9]))
        f=TrackFilter()
        boxes=[[798.7,687.7,942.5,856.4],[802.6,687.2,894.9,856.8],[823.1,691.6,894.4,855.7]]
        self.assertEqual(len(f.update([player(i,b,c) for i,b,c in zip([145,3,134],boxes,[.74,.50,.18])])),1)
        # The nearby goalkeeper and two defenders must remain separate.
        boxes=[[904.3,416.2,951.0,531.0],[961.6,440.2,1016.5,588.4],[901.8,459.8,1009.4,635.0]]
        self.assertEqual(len(TrackFilter().update([player(i,b) for i,b in enumerate(boxes)])),3)

    def test_three_boxes_on_same_player_keep_best(self):
        f=TrackFilter()
        result=f.update([player(1,[100,100,180,300],.5),player(2,[105,101,185,300],.9),player(3,[98,99,179,301],.2)])
        self.assertEqual([p['id'] for p in result],[2])
        self.assertEqual(f.suppressed,2)

    def test_adjacent_players_remain(self):
        f=TrackFilter()
        self.assertEqual(len(f.update([player(1,[100,100,180,300]),player(2,[145,100,225,300])])),2)

    def test_occluded_players_with_different_feet_remain(self):
        self.assertFalse(duplicate_boxes([100,100,180,300],[100,60,180,260]))

    def test_continuity_breaks_near_tie(self):
        f=TrackFilter();f.update([player(1,[100,100,180,300],.7)])
        result=f.update([player(1,[100,100,180,300],.7),player(2,[101,100,181,300],.75)])
        self.assertEqual(result[0]['id'],1)

    def test_empty_frame_and_degenerate_box(self):
        f=TrackFilter();self.assertEqual(f.update([]),[])
        self.assertFalse(duplicate_boxes([1,1,1,1],[1,1,1,1]))
