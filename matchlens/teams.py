import numpy as np


def kit_feature(frame, box):
    """Soft color histogram from the torso, with local background suppression.

    All hues are allowed. Side strips estimate background appearance; only
    outer torso pixels are downweighted when they match that background.
    This is a crop heuristic, not a learned player segmentation model.
    """
    import cv2
    x1, y1, x2, y2 = map(float, box)
    height, width = frame.shape[:2]
    left, right = max(0, int(x1+.16*(x2-x1))), min(width, int(x2-.16*(x2-x1)))
    top, bottom = max(0, int(y1+.18*(y2-y1))), min(height, int(y1+.48*(y2-y1)))
    crop = frame[top:bottom, left:right]
    if crop.size < 24:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(float)
    h,s,v = (hsv[:,:,i] for i in range(3))
    yy,xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
    horizontal = (xx+.5)/crop.shape[1]
    weights = np.exp(-.5*((horizontal-.5)/.25)**2)
    strips = [frame[top:bottom,max(0,int(x1-.15*(x2-x1))):max(0,int(x1))],
              frame[top:bottom,min(width,int(x2)):min(width,int(x2+.15*(x2-x1)))]]
    pixels = [strip.reshape(-1,3) for strip in strips if strip.size]
    if pixels:
        bg = np.concatenate(pixels)
        lab_bg = cv2.cvtColor(bg.reshape(-1,1,3),cv2.COLOR_BGR2LAB).reshape(-1,3).astype(float)
        center = np.median(lab_bg,axis=0)
        # Suppress only a coherent surrounding color, and retain the central
        # torso even when a jersey happens to match the pitch.
        if np.median(np.linalg.norm(lab_bg-center,axis=1)) < 28:
            lab = cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(float)
            match = np.linalg.norm(lab-center,axis=2)<22
            edge = (horizontal<.25)|(horizontal>.75)
            weights[match & edge] *= .15
    hist = np.zeros(15)
    chroma = np.clip((s-20)/45,0,1)*np.clip((v-25)/40,0,1)
    hue = h/15
    base = np.floor(hue).astype(int)%12
    fraction = hue-np.floor(hue)
    for offset,factor in [(0,1-fraction),(1,fraction)]:
        hist[:12] += np.bincount(((base+offset)%12).ravel(),
                                weights=(weights*chroma*factor).ravel(),minlength=12)
    # Neutral bins retain the information in white and black kits.
    neutral = np.where(v<65,0,np.where(v<170,1,2))
    hist[12:] = np.bincount(neutral.ravel(),weights=(weights*(1-chroma)).ravel(),minlength=3)
    return np.sqrt(hist/hist.sum())


class Teams:
    """Learn two supported kit groups; leave outliers unassigned."""
    def __init__(self):
        self.centers = None
        self.colors = {}
        self.counts = {}
        self.unavailable = set()

    def update(self, observations):
        for ident, color in observations:
            if color is None:
                self.unavailable.add(ident)
                continue
            self.unavailable.discard(ident)
            color = np.asarray(color, float)
            self.colors[ident] = .8*self.colors.get(ident, color) + .2*color
            self.counts[ident] = self.counts.get(ident, 0)+1
        if self.centers is not None:
            # Adapt slowly using only confident, currently visible groups.
            # Fixed center indices preserve Team A/B identity.
            for team in (0,1):
                members = [self.colors[k] for k,_ in observations if self.label(k)==team]
                if len(members)>=2:
                    self.centers[team] = .99*self.centers[team]+.01*np.mean(members,axis=0)
            return
        data = np.array([v for k,v in self.colors.items() if self.counts[k] >= 5])
        if len(data) < 4:
            return
        best = None
        # Equal weight per track prevents one persistent goalkeeper from
        # dominating the training data. Both teams need independent support.
        for i in range(len(data)):
            for j in range(i):
                if np.linalg.norm(data[i]-data[j]) < .65:
                    continue
                centers = data[[i,j]]
                d = np.linalg.norm(data[:,None]-centers, axis=2)
                labels = d.argmin(1)
                inlier = d.min(1) < .5
                counts = [np.sum(inlier & (labels==k)) for k in (0,1)]
                if min(counts) < 2:
                    continue
                score = (sum(counts), min(counts), -float(d.min(1)[inlier].sum()))
                if best is None or score > best[0]:
                    refined = np.array([data[inlier & (labels==k)].mean(0) for k in (0,1)])
                    best = score, refined
        if best is not None:
            self.centers = best[1]

    def label(self, ident):
        if self.centers is None or ident not in self.colors or ident in self.unavailable:
            return -1
        d = np.linalg.norm(self.centers-self.colors[ident], axis=1)
        order = np.argsort(d)
        return int(order[0]) if d[order[0]] < .55 and d[order[1]]-d[order[0]] > .15 else -1
