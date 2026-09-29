"""Conservative suppression of duplicate tracker outputs within one frame."""
import numpy as np


def duplicate_boxes(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sizes = np.array([a[2:]-a[:2], b[2:]-b[:2]])
    if np.any(sizes <= 0):
        return False
    intersection = np.maximum(0, np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    areas = sizes.prod(axis=1)
    iou = intersection/(areas.sum()-intersection)
    feet = np.array([[(a[0]+a[2])/2,a[3]],[(b[0]+b[2])/2,b[3]]])
    # Foot agreement prevents merging vertically occluded players merely
    # because one rectangle contains much of the other.
    difference = np.abs(feet[0]-feet[1])
    containment = intersection/areas.min()
    return bool((iou >= .60 or containment >= .80) and difference[0] <= .30*sizes[:,0].max()
                and difference[1] <= .12*sizes[:,1].min())


class TrackFilter:
    def __init__(self):
        self.previous = set()
        self.suppressed = 0

    def update(self, players):
        # Modest continuity preference; a much stronger new detection wins.
        ranked = sorted(players, key=lambda p: (p['confidence']+.1*(p['id'] in self.previous),
                                                p['confidence'], -p['id']), reverse=True)
        kept = []
        for player in ranked:
            if any(duplicate_boxes(player['box'], other['box']) for other in kept):
                self.suppressed += 1
            else:
                kept.append(player)
        self.previous = {p['id'] for p in kept}
        return sorted(kept, key=lambda p:p['id'])


