"""Board pose from the silkscreen itself: match SIFT features against a reference whose points have known board-plane
coordinates, then solve PnP from hundreds of 3D-2D pairs. No colour mask, so glare and wood can't cut the board.

The reference is built once from a frame whose pose is already good (the colour-segmentation PnP, corner fit < ~1.5 px):
every SIFT keypoint inside the board outline gets its board-plane (x, y, 0) mm from the ray-plane intersection.
    python feature_pose.py build FRAME.jpg [--flip]     writes board_features.npz
    python feature_pose.py test FRAME.jpg               solves that frame from the features
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pcb_pnp import solve_frame
from pnp_test import load_intrinsics, to_board

HERE = Path(__file__).resolve().parent
NPZ = HERE / "board_features.npz"


def _sift():
    return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)


class BoardFeatures:
    def __init__(self, desc, pts3d):
        self.desc, self.pts3d = desc.astype(np.float32), pts3d.astype(np.float32)
        self.sift = _sift()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    @classmethod
    def load(cls, path=NPZ):
        d = np.load(path)
        return cls(d["desc"], d["pts3d"])

    def save(self, path=NPZ):
        np.savez_compressed(path, desc=self.desc, pts3d=self.pts3d)

    @classmethod
    def build(cls, frame, rvec, tvec, K, dist, width, height):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = _sift().detectAndCompute(gray, None)
        keep, pts = [], []
        for i, kp in enumerate(kps):
            p, _ = to_board(kp.pt, K, dist, rvec, tvec)
            if 1.0 <= p[0] <= width - 1.0 and 1.0 <= p[1] <= height - 1.0:   # inside the outline, not on its edge
                keep.append(i)
                pts.append([p[0], p[1], 0.0])
        return cls(desc[keep], np.array(pts))

    def solve(self, frame, K, dist, min_inliers=15):
        """Returns (rms_px, rvec, tvec, n_inliers) or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kps) < min_inliers:
            return None
        good = []
        for m, n in self.matcher.knnMatch(desc, self.desc, k=2):
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < min_inliers:
            return None
        img = np.array([kps[m.queryIdx].pt for m in good], np.float32)
        obj = self.pts3d[[m.trainIdx for m in good]]
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, dist, iterationsCount=300, reprojectionError=4.0,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < min_inliers:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)
        rep = cv2.projectPoints(obj[inl], rvec, tvec, K, dist)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((rep - img[inl]) ** 2, axis=1))))
        return rms, rvec, tvec, len(inl)


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    K, dist, _ = load_intrinsics(HERE / "camera_intrinsics.json")
    geo = json.loads((HERE / "pcb_geometry.json").read_text(encoding="utf-8-sig"))
    frame = cv2.imread(path)
    if mode == "build":
        W, H = geo["width"], geo["height"]
        obj = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
        res = solve_frame(frame, obj, K, dist, None, "--flip" in sys.argv)
        rms, k, rvec, tvec = res[0]
        print(f"reference pose from colour segmentation: corner fit {rms:.2f} px")
        if rms > 1.5:
            raise SystemExit("that frame's pose isn't good enough to build the reference from")
        bf = BoardFeatures.build(frame, rvec, tvec, K, dist, W, H)
        bf.save()
        print(f"saved {len(bf.pts3d)} board features to {NPZ.name}")
    else:
        bf = BoardFeatures.load()
        r = bf.solve(frame, K, dist)
        if r is None:
            print("no pose from features")
        else:
            rms, rvec, tvec, n = r
            R = cv2.Rodrigues(rvec)[0]
            cam = -R.T @ tvec.ravel()
            print(f"features: {n} inliers, rms {rms:.2f} px, camera {abs(cam[2]):.1f} mm up, tilt "
                  f"{np.degrees(np.arccos(min(1, abs(R[2, 2])))):.1f} deg")
