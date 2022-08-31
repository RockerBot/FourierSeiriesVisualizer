import cv2 as cv
import numpy as np
import sys, os, time, math
from datetime import datetime
import pygame, threading

# not necessary TODO delete /*
sys.path.append(os.path.abspath("D:/DATA/PYCHARM PROJECTS"))
from FluffStuff.my_helpables import ProgressBar
# TODO delete */


zoom_factor = 200
mid = [0, 0]
ofst = [0, 0]

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


def process_pic(img, key, res_dict):
    total_on_pixels = 0
    curr_on_pixels = 0

    priority_dict = {}  # priority value for each position (x, y)
    prev_dict = {}  # prev node for each position (x, y)
    pqueue = []  # the priority queue will have positions(x, y)

    for row in img: total_on_pixels += sum(row)
    total_on_pixels /= 255

    progBar = ProgressBar(
        tags=ProgressBar.COLOUR | ProgressBar.SLASH | ProgressBar.BLOCK | ProgressBar.PERCENT | ProgressBar.RELOAD
    )

    start_pt = find_strt(img)
    priority_dict[start_pt] = 0
    curr_on_pixels += 1

    q_push(pqueue, start_pt, p_dict=priority_dict)

    while total_on_pixels != curr_on_pixels:
        print(progBar.update(curr_on_pixels, total=total_on_pixels), end='')

        curr = q_pop(pqueue, p_dict=priority_dict)

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
    res_dict[key] = img


def get_pic():
    wbcm = cv.VideoCapture(0)
    blk_sz = 11
    tresh1 = 150
    tresh2 = 175
    canny = False
    while True:
        cam_img = wbcm.read()[1]
        if canny:
            blue = cv.Canny(np.array(cam_img[:, :, 0]), tresh1, tresh2)
            green = cv.Canny(np.array(cam_img[:, :, 1]), tresh1, tresh2)
            red = cv.Canny(np.array(cam_img[:, :, 2]), tresh1, tresh2)
        else:
            fltr = cv.bilateralFilter(np.array(cam_img[:, :, 0]), 10, 15, 15)
            blue = cv.adaptiveThreshold(fltr, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blk_sz, 5)
            fltr = cv.bilateralFilter(np.array(cam_img[:, :, 1]), 10, 15, 15)
            green = cv.adaptiveThreshold(fltr, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blk_sz, 5)
            fltr = cv.bilateralFilter(np.array(cam_img[:, :, 2]), 10, 15, 15)
            red = cv.adaptiveThreshold(fltr, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blk_sz, 5)

        img = np.stack((blue, green, red), axis=2)
        cv.imshow('BORDERS', img)
        # cv.imshow('BORDERS_R', red)
        # cv.imshow('BORDERS_G', green)
        # cv.imshow('BORDERS_B', blue)

        if (key := cv.waitKey(1) & 0xFF) == ord(' '):
            wbcm.release()
            cv.destroyAllWindows()

            img_h, img_w = len(red)//2, len(red[0])//2

            # cuts the image width and height by half
            red = np.array([
                [any((red[2*i][2*j], red[2*i+1][2*j], red[2*i][2*j+1], red[2*i+1][2*j+1])) * 255
                 for j in range(img_w)] for i in range(img_h)
            ]).astype(np.uint8)
            green = np.array([
                [any((green[2*i][2*j], green[2*i+1][2*j], green[2*i][2*j+1], green[2*i+1][2*j+1])) * 255
                 for j in range(img_w)] for i in range(img_h)
            ]).astype(np.uint8)
            blue = np.array([
                [any((blue[2*i][2*j], blue[2*i+1][2*j], blue[2*i][2*j+1], blue[2*i+1][2*j+1])) * 255
                 for j in range(img_w)] for i in range(img_h)
            ]).astype(np.uint8)

            return red, green, blue

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

        print(f'1:{tresh1} 2:{tresh2}' if canny else f'{blk_sz=}')


def get_path(img, key, res_dict, /):
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
    res_dict[key] = path


def thread_func(func, imgs, res_dict):
    t1 = threading.Thread(target=func, args=(imgs[0], 'red', res_dict))
    t2 = threading.Thread(target=func, args=(imgs[1], 'green', res_dict))
    t3 = threading.Thread(target=func, args=(imgs[2], 'blue', res_dict))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()


def place(x, y): return x*zoom_factor/100+mid[0]+ofst[0], y*zoom_factor/100+mid[1]+ofst[1]
def place_pt(x): return x[0]*zoom_factor/100+mid[0]+ofst[0], x[1]*zoom_factor/100+mid[1]+ofst[1]


def dft(x: list[tuple], key, res_dict):
    N = len(x)
    X = []
    progBar = ProgressBar(tags=ProgressBar.SLASH | ProgressBar.RELOAD | ProgressBar.COLOUR, total=N)

    for k in range(N):
        print(progBar.update(k), end="")
        c_num = 0 + 0j
        for n in range(N):
            phi = (2 * math.pi * k * n) / N
            c_num += complex(*x[n]) * complex(cos(phi), -sin(phi))
        xn, yn = c_num.real/N, c_num.imag/N
        X.append(((xn, yn), (xn**2+yn**2)**0.5, k, math.atan2(yn, xn)))  # (x, y), amplitude, frequency, phase
    res_dict[key] = X


def epi_cycles(fourier: list, hud_screen, t, must_draw: bool, rotation=0.0) -> tuple:
    x = y = 0
    for i, val in enumerate(fourier):
        past_pt = x, y
        freq = val[2]
        r = val[1]
        phase = val[3]
        x += r * math.cos(freq * t + phase + rotation)
        y += r * math.sin(freq * t + phase + rotation)
        if must_draw: circ(hud_screen, (255, 255, 255, 55), place_pt(past_pt), r*zoom_factor/100, 1)
        if must_draw: line(hud_screen, (0, 255, 0, 55), place_pt(past_pt), place(x, y))
    return x, y


def do_fourier_draw(height, width, paths):
    global zoom_factor, mid, ofst

    Tau = 2 * math.pi

    img_created = [False]*3
    w_key = False
    a_key = False
    s_key = False
    d_key = False
    dn_arrow = False
    up_arrow = False
    rht_arrow = False
    lft_arrow = False
    hud_on = True
    r_key = True
    g_key = True
    b_key = True

    t_r = t_g = t_b = 0
    background = (34, 34, 34, 255)
    transparency = 85

    fps = 60 * 20

    y_set, x_set = height / 2, width / 2
    in_put_red = [(x - x_set, y - y_set) for i, (x, y) in enumerate(paths['red']) if i % 5 == 0]
    in_put_green = [(x - x_set, y - y_set) for i, (x, y) in enumerate(paths['green']) if i % 5 == 0]
    in_put_blue = [(x - x_set, y - y_set) for i, (x, y) in enumerate(paths['blue']) if i % 5 == 0]
    max_l_r, max_l_g, max_l_b = len(in_put_red), len(in_put_green), len(in_put_blue)

    path_r, path_g, path_b = [], [], []

    fouriers = {}
    thread_func(dft, (in_put_red, in_put_green, in_put_blue), fouriers)
    fourier_red, fourier_green, fourier_blue = fouriers['red'], fouriers['green'], fouriers['blue']
    fourier_red.sort(key=lambda val: val[1], reverse=True)
    fourier_green.sort(key=lambda val: val[1], reverse=True)
    fourier_blue.sort(key=lambda val: val[1], reverse=True)
    dt_r, dt_g, dt_b = Tau / len(fourier_red), Tau / len(fourier_green), Tau / len(fourier_blue)

    pygame.init()
    my_screen = pygame.display.set_mode((width * 2, height * 2), pygame.RESIZABLE)

    print("DRAWING...")
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE:
                    path_r, path_g, path_b = [], [], []
                    img_created = [False]*3
                elif event.key == pygame.K_w: w_key = True
                elif event.key == pygame.K_a: a_key = True
                elif event.key == pygame.K_d: d_key = True
                elif event.key == pygame.K_s: s_key = True
                elif event.key == pygame.K_r: r_key = not r_key
                elif event.key == pygame.K_g: g_key = not g_key
                elif event.key == pygame.K_b: b_key = not b_key
                elif event.key == pygame.K_h: hud_on = not hud_on
                elif event.key == pygame.K_UP:
                    up_arrow = True
                    print(f'{fps=}')
                elif event.key == pygame.K_DOWN:
                    dn_arrow = True
                    print(f'{fps=}')
                elif event.key == pygame.K_RIGHT:
                    rht_arrow = True
                    print(f'{transparency=}/255')
                elif event.key == pygame.K_LEFT:
                    lft_arrow = True
                    print(f'{transparency=}/255')
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    dn_arrow = False
                    print(f'{fps=}')
                elif event.key == pygame.K_UP:
                    up_arrow = False
                    print(f'{fps=}')
                elif event.key == pygame.K_RIGHT:
                    rht_arrow = False
                    print(f'{transparency=}/255')
                elif event.key == pygame.K_LEFT:
                    lft_arrow = False
                    print(f'{transparency=}/255')
                elif event.key == pygame.K_w: w_key = False
                elif event.key == pygame.K_a: a_key = False
                elif event.key == pygame.K_d: d_key = False
                elif event.key == pygame.K_s: s_key = False
            elif event.type == pygame.MOUSEWHEEL: zoom_factor = max(0.01, zoom_factor + event.y)

        ofst[1] += s_key - w_key
        ofst[0] += d_key - a_key
        fps = max(1, fps + up_arrow - dn_arrow)
        transparency = min(255, max(1, transparency + rht_arrow - lft_arrow))

        mid = [i // 2 for i in my_screen.get_size()]

        red_screen = pygame.Surface(my_screen.get_size(), pygame.SRCALPHA)
        green_screen = pygame.Surface(my_screen.get_size(), pygame.SRCALPHA)
        blue_screen = pygame.Surface(my_screen.get_size(), pygame.SRCALPHA)
        hud_screen = pygame.Surface(my_screen.get_size(), pygame.SRCALPHA)
        my_screen.fill(background)

        path_r.insert(0, epi_cycles(fourier_red, hud_screen, t_r, hud_on))
        path_g.insert(0, epi_cycles(fourier_green, hud_screen, t_g, hud_on))
        path_b.insert(0, epi_cycles(fourier_blue, hud_screen, t_b, hud_on))

        if len(path_r) > max_l_r:
            path_r.pop()
            img_created[0] = True
        if len(path_r) > 2 and r_key:
            shape(red_screen, (255, 0, 0, transparency), False, (*map(place_pt, path_r),), int(zoom_factor/100))
        if len(path_g) > max_l_g:
            path_g.pop()
            img_created[1] = True
        if len(path_g) > 2 and g_key:
            shape(green_screen, (0, 255, 0, transparency), False, (*map(place_pt, path_g),), int(zoom_factor/100))
        if len(path_b) > max_l_b:
            path_b.pop()
            img_created[2] = True
        if len(path_b) > 2 and b_key:
            shape(blue_screen, (0, 0, 255, transparency), False, (*map(place_pt, path_b),), int(zoom_factor/100))

        if hud_on: my_screen.blit(hud_screen, (0, 0))
        if r_key: my_screen.blit(red_screen, (0, 0))
        if g_key: my_screen.blit(green_screen, (0, 0))
        if b_key: my_screen.blit(blue_screen, (0, 0))

        if all(img_created):print("\rImage DONE")

        t_r += dt_r
        t_g += dt_g
        t_b += dt_b

        pygame.display.update()
        clock.tick(fps)
    cv.waitKey(1)


def main():
    red, green, blue = get_pic()

    print("processing... ")
    tttt = time.time()
    imgs = {}
    thread_func(process_pic, (red, green, blue), imgs)
    red, green, blue = imgs['red'], imgs['green'], imgs['blue']
    print('time:', time.time()-tttt)

    height, width = len(red), len(red[0])
    paths = {}
    img = np.stack((blue, green, red), axis=2)
    cv.imshow("I MG", img)
    cv.imwrite(f"SHOTS/prcss_img_{datetime.now()}_CLR__.png".replace(":", "_"), img)

    print("fourier -ing... ")
    thread_func(get_path, (red, green, blue), paths)

    print(paths)

    do_fourier_draw(height, width, paths)



if __name__ == '__main__':
    main()


