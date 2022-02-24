from random import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2

from lilia.filters import *
from lilia import perlin_noise

export_path = "output/out.svg"
draw_contours = True
draw_hatch = False
show_bitmap = True
resolution = 1024
hatch_size = 16
contour_simplify = 2

def find_edges(IM):
    print("finding edges...")
    im = np.array(IM) 
    im = cv2.GaussianBlur(im,(3,3),0)
    im = cv2.Canny(im,100,200)
    IM = Image.fromarray(im)
    return IM.point(lambda p: p > 128 and 255)  


def getdots(IM):
    print("getting contour points...")
    PX = IM.load()  # get raw pixel data
    dots = []
    w,h = IM.size
    for y in range(h):
        row = []
        for x in range(w):
            if PX[x,y] == 255:
                if len(row) > 0:
                    if x-row[-1][0] == row[-1][-1]+1: # if next Pixel at x in row is also white
                        row[-1] = (row[-1][0],row[-1][-1]+1)  # we increment second value of tuple, indicating number of subsequent points
                    else:
                        row.append((x,0))
                else:
                    row.append((x,0))
        dots.append(row)
    return dots  # dots looks like this: [[(1,v), (3,v), ...], [(1,v), (4,v), ...], ...]. Each index in first dimension indicating row y
    
def connectdots(dots):
    print("connecting contour points...")
    contours = []
    for y in range(len(dots)):
        for x,v in dots[y]:
            if y == 0:  # for first row just append all dots to countour
                contours.append([(x,y)])
            else:
                closest = -1
                cdist = 100
                for x0,v0 in dots[y-1]:  # look at any dot one row below, save the closest and its distance on the x axis
                    if abs(x0-x) < cdist:
                        cdist = abs(x0-x)
                        closest = x0

                if cdist > 3:  # if closest one row below is near - append dot
                    contours.append([(x,y)])
                else:  # look at all dots we have in our contour
                    found = 0
                    for i in range(len(contours)):
                        if contours[i][-1] == (closest,y-1):  # find closest one and expand its countour 
                            contours[i].append((x,y,))
                            found = 1
                            break
                    if found == 0:  # only triggers if no point found one row below that is less than 100 (cdist) px away
                        contours.append([(x,y)])

        for c in contours:  # remove all countours that are shorter than four points long and which did not get expanded this row
            if c[-1][1] < y - 1 and len(c)<4:
                contours.remove(c)
    return contours


def getcontours(IM,sc=2):
    print("generating contours...")
    IM = find_edges(IM)
    IM1 = IM.copy()
    IM2 = IM.rotate(-90,expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    dots1 = getdots(IM1)
    contours1 = connectdots(dots1)
    dots2 = getdots(IM2)
    contours2 = connectdots(dots2)

    for i in range(len(contours2)):
        contours2[i] = [(c[1],c[0]) for c in contours2[i]]    
    contours = contours1+contours2

    for i in range(len(contours)):
        for j in range(len(contours)):
            if len(contours[i]) > 0 and len(contours[j])>0:
                if distsum(contours[j][0],contours[i][-1]) < 8:
                    contours[i] = contours[i]+contours[j]
                    contours[j] = []

    for i in range(len(contours)):
        contours[i] = [contours[i][j] for j in range(0,len(contours[i]),8)]


    contours = [c for c in contours if len(c) > 1]

    for i in range(0,len(contours)):
        contours[i] = [(v[0]*sc,v[1]*sc) for v in contours[i]]

    for i in range(0,len(contours)):
        for j in range(0,len(contours[i])):
            contours[i][j] = int(contours[i][j][0]+10*perlin_noise.noise(i*0.5,j*0.1,1)),int(contours[i][j][1]+10*perlin_noise.noise(i*0.5,j*0.1,2))

    return contours


def hatch(IM,sc=16):
    print("hatching...")
    PX = IM.load()
    w,h = IM.size
    lg1 = []
    lg2 = []
    for x0 in range(w):
        for y0 in range(h):
            x = x0*sc
            y = y0*sc
            if PX[x0,y0] > 144:
                pass
                
            elif PX[x0,y0] > 64:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
            elif PX[x0,y0] > 16:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)])

            else:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg1.append([(x,y+sc/2+sc/4),(x+sc,y+sc/2+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)])

    lines = [lg1,lg2]
    for k in range(0,len(lines)):
        for i in range(0,len(lines[k])):
            for j in range(0,len(lines[k])):
                if lines[k][i] != [] and lines[k][j] != []:
                    if lines[k][i][-1] == lines[k][j][0]:
                        lines[k][i] = lines[k][i]+lines[k][j][1:]
                        lines[k][j] = []
        lines[k] = [l for l in lines[k] if len(l) > 0]
    lines = lines[0]+lines[1]

    for i in range(0,len(lines)):
        for j in range(0,len(lines[i])):
            lines[i][j] = int(lines[i][j][0]+sc*perlin_noise.noise(i*0.5,j*0.1,1)),int(lines[i][j][1]+sc*perlin_noise.noise(i*0.5,j*0.1,2))-j
    return lines


def distsum(*args):
    return sum([ ((args[i][0]-args[i-1][0])**2 + (args[i][1]-args[i-1][1])**2)**0.5 for i in range(1,len(args))])


def sortlines(lines):
    print("optimizing stroke sequence...")
    clines = lines[:]
    slines = [clines.pop(0)]
    while clines != []:
        x,s,r = None,1000000,False
        for l in clines:
            d = distsum(l[0],slines[-1][-1])
            dr = distsum(l[-1],slines[-1][-1])
            if d < s:
                x,s,r = l[:],d,False
            if dr < s:
                x,s,r = l[:],s,True

        clines.remove(x)
        if r == True:
            x = x[::-1]
        slines.append(x)
    return slines


def sketch(IM):
    w, h = IM.size

    IM = IM.convert("L")  # greyscale mode
    IM=ImageOps.autocontrast(IM, cutoff=10)  #increase contrast by normalizing it and cutting of 10% of maximum white/black areas 

    lines = []
    if draw_contours:
        lines += getcontours(
            IM.resize(  # resizes the image to (width, height), returns an Image object
                (resolution//contour_simplify, resolution//contour_simplify*h//w)
            ), contour_simplify
        )
    if draw_hatch:
        lines += hatch(IM.resize((resolution//hatch_size,resolution//hatch_size*h//w)),hatch_size)

    lines = sortlines(lines)
    if show_bitmap:
        disp = Image.new("RGB",(resolution,resolution*h//w),(255,255,255))
        draw = ImageDraw.Draw(disp)
        for l in lines:
            draw.line(l,(0,0,0),5)
        disp.show()
        disp.save("result.jpeg")

    f = open(export_path,'w')
    f.write(makesvg(lines))
    f.close()
    print(len(lines),"strokes.")
    print("done.")
    return lines


def makesvg(lines):
    print("generating svg file...")
    out = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'
    for l in lines:
        l = ",".join([str(p[0]*0.5)+","+str(p[1]*0.5) for p in l])
        out += '<polyline points="'+l+'" stroke="black" stroke-width="2" fill="none" />\n'
    out += '</svg>'
    return out