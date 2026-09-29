import unittest
import numpy as np
from matchlens.geometry import homography, project
from matchlens.teams import Teams
from matchlens.__main__ import Camera


class CoreTests(unittest.TestCase):
    def test_perspective_roundtrip(self):
        corners=np.array([[120,50],[890,90],[1040,700],[10,680]],float)
        pitch=np.array([[0,0],[105,0],[105,68],[0,68]],float)
        h=homography(corners,pitch)
        np.testing.assert_allclose(project(h,corners),pitch,atol=1e-8)
        points=np.array([[12,9],[42,56],[102,40]],float)
        np.testing.assert_allclose(project(h,project(np.linalg.inv(h),points)),points,atol=1e-8)

    def test_collinear_rejected(self):
        with self.assertRaises(ValueError):
            homography([[0,0],[1,1],[2,2],[3,3]],[[0,0],[1,0],[1,1],[0,1]])

    def test_infinity_is_unknown(self):
        h=np.eye(3); h[2]=[1,0,0]
        self.assertTrue(np.isnan(project(h,[[0,2]])).all())

    def test_teams_and_unknown(self):
        t=Teams()
        for _ in range(20):
            t.update([(1,[1,0,0]),(2,[0,1,0]),(4,[1,0,0]),(5,[0,1,0])])
        self.assertNotEqual(t.label(1),t.label(2))
        self.assertIn(t.label(1),(0,1))
        t.update([(3,[0,0,1])])
        self.assertEqual(t.label(3),-1)

    def test_no_separation_remains_unknown(self):
        t=Teams()
        for _ in range(40):
            t.update([(1,[100,100,100])])
        self.assertEqual(t.label(1),-1)

    def test_missing_kit_does_not_reuse_stale_label(self):
        t=Teams()
        for _ in range(6):
            t.update([(1,[1,0,0]),(2,[1,0,0]),(3,[0,1,0]),(4,[0,1,0])])
        self.assertNotEqual(t.label(1),-1)
        t.update([(1,None)])
        self.assertEqual(t.label(1),-1)

    def test_single_outlier_cannot_be_team(self):
        t=Teams()
        for _ in range(10):
            t.update([(1,[1,0,0]),(2,[1,0,0]),(3,[1,0,0]),(4,[0,1,0])])
        self.assertIsNone(t.centers)

    def test_camera_translation(self):
        import cv2
        rng=np.random.default_rng(7)
        frame=rng.integers(0,256,(300,500,3),dtype=np.uint8)
        moved=cv2.warpAffine(frame,np.float32([[1,0,5],[0,1,3]]),(500,300))
        c=Camera(np.eye(3)); c.update(frame,[])
        h=c.update(moved,[])
        self.assertIsNotNone(h)
        np.testing.assert_allclose(project(h,[[205,103]]),[[200,100]],atol=.3)

    def test_lost_camera_stays_unavailable(self):
        c=Camera(np.eye(3)); black=np.zeros((200,300,3),np.uint8)
        c.update(black,[])
        self.assertIsNone(c.update(black,[]))
        self.assertIsNone(c.update(black,[]))


if __name__=='__main__':
    unittest.main()
