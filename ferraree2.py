import timeit

import cv2 as cv
import numpy as np
import sys, os, time, math
from datetime import datetime
from colorama import Style, Fore, Back
import pygame

# not necessary TODO delete /*
sys.path.append(os.path.abspath("D:/DATA/PYCHARM PROJECTS"))
import FluffStuff.my_helpables as my_hlp
# TODO delete */



got_pic = False
zoom_factor = 200
t=0
mid = [0, 0]
ofst = [0, 0]
X, Y = 0, 0
x, y = 0, 0
dn_arrow = False
up_arrow = False
w_key = False
a_key = False
s_key = False
d_key = False

cos = math.cos
sin = math.sin
circ = pygame.draw.circle
line = pygame.draw.line
shape = pygame.draw.lines


def find_strt(mtrx):
    h, w = len(mtrx), len(mtrx[0])
    rndx = cndx = 0
    while cndx < h and rndx < w:
        for i in range(rndx, w):
            if mtrx[cndx][i]: return i, cndx
        cndx += 1
        for i in range(cndx, h):
            if mtrx[i][w - 1]: return w - 1, i
        w -= 1
        if cndx < h:
            for i in range(w - 1, rndx - 1, -1):
                if mtrx[h - 1][i]: return i, h - 1
            h -= 1
        if rndx < w:
            for i in range(h - 1, cndx - 1, -1):
                if mtrx[i][rndx]: return rndx, i
            rndx += 1
    raise ValueError('no pixels')


def q_heapify(arr, n, i, *, p_dict):
    smallest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and p_dict[arr[smallest]] > p_dict[arr[l]]: smallest = l
    if r < n and p_dict[arr[smallest]] > p_dict[arr[r]]: smallest = r
    if smallest != i:
        arr[i], arr[smallest] = arr[smallest], arr[i]
        q_heapify(arr, n, smallest, p_dict=p_dict)


def q_push(arr, pos: tuple, *, p_dict):
    arr.append(pos)
    if n := len(arr):
        for i in range(n // 2 - 1, -1, -1): q_heapify(arr, n, i, p_dict=p_dict)


def q_pop(arr, *, p_dict):
    res = arr[0]
    if len(arr) != 1: arr[0] = arr.pop()
    else: arr.pop()
    if n := len(arr):
        for i in range(n // 2 - 1, -1, -1): q_heapify(arr, n, i, p_dict=p_dict)
    return res


def get_neighbours(mtrx, pos) -> list:
    h, w = len(mtrx), len(mtrx[0])
    nbrs = []
    for dx, dy in (0, -1), (-1, 0), (0, 1), (1, 0):
        x, y = dx + pos[0], dy + pos[1]
        if h > y >= 0 and w > x >= 0: nbrs.append((x, y))
    return nbrs


def is_pixel_on(mtrx, pos) -> bool: return mtrx[pos[1]][pos[0]]


def process_pic(img):
    img_width = len(img[0])
    img_height = len(img)
    print(img_height,img_width)

    total_on_pixels = 0
    curr_on_pixels = 0

    priority_dict = {}  # priority value for each position (x, y)
    prev_dict = {}  # prev node for each position (x, y)
    pqueue = []  # the priority queue will have positions(x, y)

    for row in img: total_on_pixels += sum(row)
    total_on_pixels /= 255

    progBar = my_hlp.ProgressBar(tags=my_hlp.ProgressBar.COLOUR | my_hlp.ProgressBar.SLASH | my_hlp.ProgressBar.BLOCK
                                 | my_hlp.ProgressBar.PERCENT | my_hlp.ProgressBar.RELOAD)  # TODO delete

    start_pt = find_strt(img)
    priority_dict[start_pt] = 0
    curr_on_pixels += 1

    q_push(pqueue, start_pt, p_dict=priority_dict)

    while total_on_pixels != curr_on_pixels:
        print(progBar.update(curr_on_pixels, total=total_on_pixels), end='')

        curr = q_pop(pqueue, p_dict=priority_dict)
        # track.mtrx_log(img)  # TODO delete

        if is_pixel_on(img, curr): curr_on_pixels += 1
        for nbr in get_neighbours(img, curr):
            if (prrty := priority_dict.get(nbr)) is None or prrty > priority_dict[curr] + 1:
                if prrty is None and is_pixel_on(img, nbr):
                    priority_dict[nbr] = 0
                    prev_dict[nbr] = curr
                    q_push(pqueue, nbr, p_dict=priority_dict)
                    while not is_pixel_on(img, (nbr := prev_dict[nbr])):
                        img[nbr[1]][nbr[0]] = 255
                        total_on_pixels += 1
                        priority_dict[nbr] = 0
                        q_push(pqueue, nbr, p_dict=priority_dict)
                else:
                    priority_dict[nbr] = priority_dict[curr] + 1
                    prev_dict[nbr] = curr
                    q_push(pqueue, nbr, p_dict=priority_dict)
    return img


def get_pic():
    progBar = my_hlp.ProgressBar(tags=my_hlp.ProgressBar.COLOUR | my_hlp.ProgressBar.BLOCK | my_hlp.ProgressBar.RELOAD,
                                 # TODO delete
                                 total=(prev := 1), clr=(None, 0.5, 0.5, None))  # TODO delete
    wbcm = cv.VideoCapture(0)
    blk_sz = 11
    tresh1 = 150
    tresh2 = 175
    canny = False
    while True:
        if canny: img = cv.Canny(cv.cvtColor(wbcm.read()[1], cv.COLOR_BGR2GRAY), tresh1, tresh2)
        else:
            fltr = cv.bilateralFilter(cv.cvtColor(wbcm.read()[1], cv.COLOR_BGR2GRAY), 10, 15, 15)
            img = cv.adaptiveThreshold(fltr, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blk_sz, 5)
        cv.imshow('BORDERS', img)

        if (key := cv.waitKey(1) & 0xFF) == ord(' '):
            wbcm.release()
            cv.destroyAllWindows()

            img_h, img_w = len(img)//2, len(img[0])//2
            temp_img = [
                [any((img[2*i][2*j], img[2*i+1][2*j], img[2*i][2*j+1], img[2*i+1][2*j+1])) * 255
                 for j in range(img_w)] for i in range(img_h)]
            img = np.array(temp_img)
            img = img.astype(np.uint8)
            return img
        elif key == ord('w'):
            if canny: tresh1 += 1
            else: blk_sz += 2
        elif key == ord('a') and canny: tresh2 += 1
        elif key == ord('s'):
            if canny: tresh1 -= 1
            else: blk_sz -= 2
            if blk_sz < 3: blk_sz = 3
        elif key == ord('d') and canny: tresh2 -= 1
        elif key == ord('r'): canny = not canny
        if canny: print(f'1:{tresh1} 2:{tresh2}')
        else: print(f'{blk_sz=}')


def get_path(img, /):
    path = []
    stack = []
    pt = find_strt(img)
    visited = {pt}
    path.append(pt)
    stack.append(pt)
    while len(stack):
        for nbr in get_neighbours(img, pt):
            if nbr in visited or not is_pixel_on(img, nbr): continue
            visited.add(pt:=nbr)
            path.append(pt)
            stack.append(pt)
            break
        else:
            stack.pop()
            if len(stack)>0: path.append(pt := stack[-1])
    return path


def place(x, y): return x*zoom_factor/100+mid[0]+ofst[0], y*zoom_factor/100+mid[1]+ofst[1]
def place_pt(x): return x[0]*zoom_factor/100+mid[0]+ofst[0], x[1]*zoom_factor/100+mid[1]+ofst[1]


def dft(x: list[tuple]):
    N = len(x)
    X = []
    print(N)
    progBar = my_hlp.ProgressBar(tags=my_hlp.ProgressBar.SLASH | my_hlp.ProgressBar.RELOAD | my_hlp.ProgressBar.COLOUR, total=N)

    for k in range(N):
        print(progBar.update(k), end="")
        c_num = 0 + 0j
        for n in range(N):
            phi = (2 * math.pi * k * n) / N
            c_num += complex(*x[n]) * complex(cos(phi), -sin(phi))
        xn, yn = c_num.real/N, c_num.imag/N
        X.append(((xn, yn), (xn**2+yn**2)**0.5, k, math.atan2(yn, xn)))  # (x, y), amplitude, frequency, phase
    return X


def epi_cycles(fourier: list, my_screen, rotation=0.0):
    global x, y
    ratio = 200/len(fourier)
    for i, val in enumerate(fourier):
        past_pt = x, y
        freq = val[2]
        r = val[1]
        phase = val[3]
        x += r * math.cos(freq * t + phase + rotation)
        y += r * math.sin(freq * t + phase + rotation)
        circ(my_screen, (ratio*i+55,)*3, place_pt(past_pt), r*zoom_factor/100, 1)
        line(my_screen, 'green', place_pt(past_pt), place(x, y))


def do_fourier_draw(img, path):
    global t, zoom_factor, mid, x, y, w_key, s_key, a_key, d_key, ofst
    background = (34, 34, 34, 255)

    fps = 60 * 20
    fps_ofst = 0

    y_set, x_set = len(img) / 2, len(img[0]) / 2
    in_put = [(x - x_set, y - y_set) for i, (x, y) in enumerate(path) if i % 5 == 0]
    max_l = len(in_put)
    PATH = []

    fourier = dft(in_put)

    dt = (2 * math.pi) / len(fourier)
    pygame.init()
    my_screen = pygame.display.set_mode((len(img[0]) * 2, len(img) * 2), pygame.RESIZABLE)

    print("DRAWING")
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_r: PATH = []
                elif event.key == pygame.K_w: w_key = True
                elif event.key == pygame.K_a: a_key = True
                elif event.key == pygame.K_d: d_key = True
                elif event.key == pygame.K_s: s_key = True
                elif event.key == pygame.K_UP:
                    fps_ofst = 1
                    print(f'{fps=}')
                elif event.key == pygame.K_DOWN:
                    fps_ofst = -1
                    print(f'{fps=}')
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                    fps_ofst = 0
                    print(f'{fps=}')
                elif event.key == pygame.K_w: w_key = False
                elif event.key == pygame.K_a: a_key = False
                elif event.key == pygame.K_d: d_key = False
                elif event.key == pygame.K_s: s_key = False
            elif event.type == pygame.MOUSEWHEEL: zoom_factor = max(0.01, zoom_factor + event.y)

        ofst[1] -= w_key
        ofst[1] += s_key
        ofst[0] += d_key
        ofst[0] -= a_key
        fps = max(1, fps + fps_ofst)
        mid = [i // 2 for i in my_screen.get_size()]
        my_screen.fill(background)

        x = y = 0
        fourier.sort(key=lambda val: val[1], reverse=True)
        epi_cycles(fourier, my_screen)
        PATH.insert(0, (x, y))
        if len(PATH) > max_l: PATH.pop()
        if len(PATH) > 2: shape(my_screen, (255, 255, 255, 0.2), False, (*map(place_pt, PATH),), 2)

        t += dt
        pygame.display.update()
        clock.tick(fps)
    cv.waitKey(1)


def main():
    img = get_pic()

    print("processing... ")
    tttt =time.time()
    img = process_pic(img)
    print('time:', time.time()-tttt)

    cv.imshow("I MG", img)
    cv.imwrite(f"SHOTS/prcss_img_{datetime.now()}__.png".replace(":", "_"), img)

    print("fourier -ing... ")
    path = get_path(img)
    print(path)
    do_fourier_draw(img, path)



if __name__ == '__main__':
    main()


