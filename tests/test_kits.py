import unittest
import numpy as np
from matchlens.teams import kit_feature, Teams


def jersey(color, background=(30,100,40), stripes=None):
    image=np.full((120,100,3),background,np.uint8)
    image[20:65,30:70]=color
    if stripes is not None:
        image[20:65,30:40]=stripes
        image[20:65,50:60]=stripes
    return kit_feature(image,[20,0,80,120])


class KitTests(unittest.TestCase):
    def test_green_is_a_valid_kit(self):
        green=jersey((25,140,30))
        red=jersey((25,25,200))
        self.assertIsNotNone(green)
        self.assertGreater(np.linalg.norm(green-red),.65)

    def test_white_and_black_are_distinct(self):
        self.assertGreater(np.linalg.norm(jersey((230,230,230))-jersey((25,25,25))),.65)

    def test_striped_kit_retains_red(self):
        striped=jersey((20,20,200),stripes=(230,230,230))
        blue=jersey((220,155,100))
        self.assertGreater(np.linalg.norm(striped-blue),.65)

    def test_local_background_does_not_remove_green_center(self):
        feature=jersey((30,130,40),background=(30,130,40))
        self.assertIsNotNone(feature)
        self.assertAlmostEqual(float(np.linalg.norm(feature)),1)

    def test_automatic_green_white_groups_and_outlier(self):
        t=Teams()
        green,white,red=jersey((30,160,40)),jersey((230,230,230)),jersey((20,20,220))
        for _ in range(12):
            t.update([(1,green),(2,green),(3,white),(4,white),(5,red)])
        self.assertIn(t.label(1),(0,1))
        self.assertIn(t.label(3),(0,1))
        self.assertNotEqual(t.label(1),t.label(3))
        self.assertEqual(t.label(5),-1)
        labels=[t.label(i) for i in range(1,5)]
        for _ in range(30):
            t.update([(4,white),(3,white),(2,green),(1,green)])
        self.assertEqual(labels,[t.label(i) for i in range(1,5)])
