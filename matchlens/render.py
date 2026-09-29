import cv2
import numpy as np

COLORS = [(245, 183, 45), (112, 224, 94), (145, 145, 145)]


def dashboard(frame, players, number, fps, status, length=105., width=68.):
    canvas = np.full((800, 1440, 3), (22, 19, 15), np.uint8)
    def text(s, pos, size=.6, color=(220, 226, 222)):
        cv2.putText(canvas, s, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    text('MATCHLENS / AI', (28, 40), .9, (125, 240, 186))
    text('BROADCAST TO TACTICAL VIEW', (28, 70), .48)
    text(f'FRAME {number:06d}   |   {number/fps:06.1f}s', (1000, 42), .6)
    image = frame.copy()
    for p in players:
        x1, y1, x2, y2 = map(int, p['box'])
        color = COLORS[p['team']] if p['team'] >= 0 else COLORS[2]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"#{p['id']}", (x1, max(12,y1-5)), 0, .45, color, 1)
    scale = min(920/image.shape[1], 600/image.shape[0])
    image = cv2.resize(image, None, fx=scale, fy=scale)
    canvas[110:110+image.shape[0], 24:24+image.shape[1]] = image
    text('TACTICAL VIEW', (990, 120), .7)
    text(f'{length:g} x {width:g} m', (990, 146), .45)
    ox, oy, pw, ph = 990, 195, 420, int(420*width/length)
    def xy(x,y):
        return int(ox+x/length*pw), int(oy+y/width*ph)
    for i in range(10):
        cv2.rectangle(canvas, (ox+i*42,oy), (ox+(i+1)*42,oy+ph), (34+4*(i%2),53+4*(i%2),29), -1)
    line = (154,175,151)
    cv2.rectangle(canvas, (ox,oy), (ox+pw,oy+ph), line, 1)
    cv2.line(canvas, xy(length/2,0), xy(length/2,width), line,1)
    cv2.circle(canvas, xy(length/2,width/2), int(9.15/length*pw), line,1)
    for x, sign in [(0,1),(length,-1)]:
        cv2.rectangle(canvas, xy(x,width/2-20.16), xy(x+sign*16.5,width/2+20.16), line,1)
    for p in players:
        x,y = p['pitch']
        if np.isfinite([x,y]).all() and 0 <= x <= length and 0 <= y <= width:
            color = COLORS[p['team']] if p['team'] >= 0 else COLORS[2]
            cv2.circle(canvas, xy(x,y), 5, color, -1, cv2.LINE_AA)
            px,py = xy(x,y)
            text(str(p['id']), (px+6,py-4), .3)
    text(status[:48], (990, 520), .45)
    for team in (0,1):
        points = np.array([p['pitch'] for p in players if p['team']==team and np.isfinite(p['pitch']).all() and 0<=p['pitch'][0]<=length and 0<=p['pitch'][1]<=width])
        text(f'TEAM {"AB"[team]} / {len(points)} mapped', (990,565+team*60), .55, COLORS[team])
        if len(points)>1:
            spread = np.ptp(points, axis=0)
            text(f'Spread {spread[0]:.1f} x {spread[1]:.1f} m', (990,588+team*60), .45)
    text('PLAYER POSITIONS / FOOT-POINT PROJECTION', (28,745), .52)
    text('Unknown kits: gray   |   Possession unavailable   |   Ball tracking: experimental / not included', (28,775), .47)
    return canvas
