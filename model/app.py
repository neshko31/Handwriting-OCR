import math
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import tensorflow as tf
from keras.models import load_model
import os
import cv2
from sklearn.cluster import KMeans

# --- FUNKCIJE ZA OBRADU SLIKE I PREPOZNAVANJE TEKSTA ---
# Ćelija u kojoj se nalaze Utility funkcije

gpsv = np.random.default_rng(seed=42)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image, color=False):
    if color:
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Slika ima 3 kanala => verovatno BGR, konvertuj u RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
        else:
            # Slika je grayscale (1 kanal), nema konverzije
            print("Upozorenje: Pokušaj prikaza grayscale slike kao RGB. Prikazujem kao grayscale.")
            plt.imshow(image, cmap='gray')
    else:
        # Uvek grayscale prikaz
        plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def image_gray(image):
    # Ako slika već ima 1 kanal (grayscale), ne radi konverziju
    if len(image.shape) == 2:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def image_bin(image_gs):                                                    # gs -> gray scale
    image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_OTSU)[1]
    return image_bin

def invert(image):
    return 255-image

def invert_bin(image):
    return cv2.bitwise_not(image)

def dilate(image):                                                          # prosiruje bele delove slike
    kernel = np.ones((3, 3))                                                # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):                                                           # smannjuje bele delove slike
    kernel = np.ones((3, 3))                                                # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def opening(image):                                                         #otvaranje = erozija + dilacija
    return erode(dilate(image))                                             #uklanjanje šuma erozijom i vraćanje originalnog oblika dilacijom

def closing(image):                                                         #zatvaranje = dilacija + erozija,
    return dilate(erode(image))                                             #zatvaranje sitnih otvora među belim pikselima

def resize_region(region):
    # If region has a channel dimension, remove it for resizing
    if len(region.shape) == 3 and region.shape[2] == 1:
        region_2d = region.squeeze(axis=2)
    else:
        region_2d = region
    
    # Resize the 2D image
    resized = cv2.resize(region_2d, (64, 64), interpolation=cv2.INTER_AREA)
    
    return resized

def matrix_to_vector(image):                                                #pretvara sliku u vektoru
    return image.flatten()

def scale_to_range(image):                                                  #skalira boje sa opsega [0, 255] na [0, 1]   
    # Convert to float32 first
    image = image.astype(np.float32)
    
    # Normalize based on the actual range, not assuming [0, 255]
    image_min = np.min(image)
    image_max = np.max(image)
    
    if image_max > 1.0:  # Assume it's in [0, 255] range
        image = image / 255.0
    # If already in [0, 1] range, don't change it
    
    return image

def rotate_image_to_normal(image, image_grayscale):
    image_edges = cv2.Canny(image_grayscale, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(image_edges, 1, math.pi / 180.0, 100, maxLineGap=5)

    angles = []

    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    
        median_angle = np.median(angles)
        image_rotated = ndimage.rotate(image, median_angle, cval=220, reshape=True)
    else:
        image_rotated = image
    return image_rotated


def order_points(pts):                                                                                 # pomoćna funkcija, biće potrebna za correct_skewed_image
    rect = np.zeros((4, 2), dtype="float32")                                                           

    s = pts.sum(axis=1)                                                                               
    rect[0] = pts[np.argmin(s)]                                                                         # top-left (najmanji zbir x + y)
    rect[2] = pts[np.argmax(s)]                                                                         # bottom-right (najveći zbir x + y)

    diff = np.diff(pts, axis=1)                                                                        
    rect[1] = pts[np.argmin(diff)]                                                                      # top-right (najmanja razlika x - y)
    rect[3] = pts[np.argmax(diff)]                                                                      # bottom-left (najveća razlika x - y)

    return rect


def correct_skewed_image(image):
    gray = image_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = image.shape[0] * image.shape[1]
    found = False

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)

        # Zakomentarisana linija zbog prevelikog outputa
        # print(f"Dužina konture: {len(approx)}, Površina: {area:.2f}, Udeo u slici: {area / image_area:.2%}")

        if len(approx) == 4 and area > 0.5 * image_area:
            screen_cnt = approx
            found = True

            # Prikaz konture za proveru
            debug_img = image.copy()
            cv2.drawContours(debug_img, [screen_cnt], -1, (0, 255, 0), 2)
            print("Pronađena validna kontura sa 4 temena.")
            display_image(debug_img, color=True)
            break

    if not found:
        # Zakomentarisana linija zbog prevelikog outputa
        # print("Nije pronađena validna kontura za ispravljanje.")
        return image  # Vraća original ako nema dovoljno dobre konture

    rect = order_points(screen_cnt.reshape(4, 2))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    print("Dimenzije transformacije:", max_width, "x", max_height)
    print("Tačke ulaza (rect):", rect)
    print("Tačke izlaza (dst):", dst)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

def select_roi_training(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        region = image_bin[y:y+h, x:x+w]
        if region is not None:
            regions_array.append([region, (x, y, w, h)])

    # Konture koje sadrze kukice
    connected_regions = [] 
    # Prolazimo kroz niz i proveravamo da li u ostatku niza postoji neki region iznad ili ispod trenutnog i ako 
    # postoji takav region, pravimo jedan veci region oko njih i dodajemo ga
    for region1, (x1, y1, w1, h1) in regions_array:
        for _region2, (x2, y2, w2, h2) in regions_array:
            # Provera da li je isti region
            if (x1, y1, w1, h1) == (x2, y2, w2, h2):
                continue
            mid_x2 = x2 + w2 // 2

            # Provera da li je region kvacica
            if (y1 >= y2 or y1 <= y2) and mid_x2 >= x1 and mid_x2 <= x1 + w1:
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2) # Desna ivica
                y_max = max(y1 + h1, y2 + h2) # Donja ivica

                # Novi kombinovani bounding box (x3, y3, w3, h3)
                x3 = x_min
                y3 = y_min
                w3 = x_max - x_min
                h3 = y_max - y_min
                region3 = image_bin[y3:y3+h3, x3:x3+w3]
                connected_regions.append([resize_region(region3), (x3, y3, w3, h3)])

    # Sad cemo proci kroz niz regions_array i videti da li se centar tih objekata nalazi unutar 
    # nekog veceg koji je u connected_regions
    # Filtiramo regione tako da na primer za slovo ž sad imamo 2 regiona u regions_array i 1 veci
    # region u connected_regions
    # Ona 2 regiona ne dodajemo u filtrirane regione dok cemo 1 veci region dodati kasnije
    # Ako se nalazi, preskacemo taj objekat, a ako ne onda ga dodajemo u nov niz
    filtered_regions = []
    for region1, (x1, y1, w1, h1) in regions_array:
        mid_x1 = x1 + w1 // 2
        mid_y1 = y1 + h1 // 2
        is_in = False
        for _region2, (x2, y2, w2, h2) in connected_regions:
            if mid_x1 >= x2 and mid_x1 <= x2 + w2 and mid_y1 >= y2 and mid_y1 <= y2 + h2:
                is_in = True
        if not is_in:
            filtered_regions.append([region1, (x1, y1, w1, h1)])
    
    # Dodajemo i one velike regione u kojem su slova sa kukicama
    for connected_region, (x1, y1, w1, h1) in connected_regions:
        is_in = False
        for region, (x2, y2, w2, h2) in filtered_regions:
            if (x1, y1, w1, h1) == (x2, y2, w2, h2):
                is_in = True
                break
        if not is_in:
            filtered_regions.append([connected_region, (x1, y1, w1, h1)])

    # Iscrtavamo sve te regione
    for _, (x, y, w, h) in filtered_regions:
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    # Sortiramo niz u kojem su elementi [region, koordinate] po velicini regiona
    filtered_regions = sorted(filtered_regions, key=lambda x: x[1][2] * x[1][3], reverse=True)
    sorted_regions = [resize_region(region[0]) for region in filtered_regions]

    if len(sorted_regions) == 0:
        return image_orig, [image_bin]
    else:
        return image_orig, sorted_regions
    
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
				# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# return only the bounding boxes that were picked
	return boxes[pick]

def select_roi_test(image_orig, image_bin):
	contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	sorted_regions = [] # lista sortiranih regiona po X osi
	regions_array = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour) # koordinate i velicina granicnog pravougaonika
		# kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
		# oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
		region = image_bin[y:y+h+1, x:x+w+1]
		regions_array.append([resize_region(region), (x, y, w, h)])

	#
	#
	# ---------- POBOLJŠAN DEO ZA SPAJANJE KVAČICA ----------
	#
	#
	
	# Konture koje sadrze kvačice - koristićemo set za brže pretraživanje
	connected_regions = [] 
	used_regions = set()  # Pratimo koje regione smo već spojili
	
	# Prvo prolazimo kroz sve regione i pronalazimo kandidate za kvačice
	for i, (region1, (x1, y1, w1, h1)) in enumerate(regions_array):
		if i in used_regions:
			continue
			
		# Lista regiona koji će biti spojeni sa trenutnim
		regions_to_merge = [i]
		
		# Pronađemo sve regione koji mogu biti deo iste kvačice
		for j, (region2, (x2, y2, w2, h2)) in enumerate(regions_array):
			if i == j or j in used_regions:
				continue
			
			# Izračunavamo centre regiona
			center_x1, center_y1 = x1 + w1 // 2, y1 + h1 // 2
			center_x2, center_y2 = x2 + w2 // 2, y2 + h2 // 2
			
			# Poboljšani uslovi za prepoznavanje kvačica:
			# 1. Horizontalno preklapanje - centri su blizu po x osi
			horizontal_overlap = abs(center_x1 - center_x2) <= max(w1, w2) * 0.7
			
			# 2. Vertikalna bliskost - regioni su jedan iznad drugog
			vertical_gap = abs(center_y1 - center_y2) - (h1 + h2) // 2
			max_vertical_gap = max(h1, h2) * 0.3  # Dozvoljavamo manji razmak
			vertical_close = vertical_gap <= max_vertical_gap
			
			# 3. Slična širina - kvačice obično imaju sličnu širinu
			width_ratio = min(w1, w2) / max(w1, w2)
			similar_width = width_ratio >= 0.4
			
			# 4. Razumna visina - ne spajamo regione koji su previše različiti po visini
			height_ratio = min(h1, h2) / max(h1, h2)
			reasonable_height = height_ratio >= 0.3
			
			# Ako su ispunjeni svi uslovi, ovaj region je kandidat za spajanje
			if horizontal_overlap and vertical_close and similar_width and reasonable_height:
				regions_to_merge.append(j)
		
		# Ako imamo više od jednog regiona za spajanje, kreiramo spojeni region
		if len(regions_to_merge) > 1:
			# Označavamo sve regione kao korišćene
			used_regions.update(regions_to_merge)
			
			# Pronalazimo granice spojenog regiona
			all_x = [regions_array[idx][1][0] for idx in regions_to_merge]
			all_y = [regions_array[idx][1][1] for idx in regions_to_merge]
			all_x_max = [regions_array[idx][1][0] + regions_array[idx][1][2] for idx in regions_to_merge]
			all_y_max = [regions_array[idx][1][1] + regions_array[idx][1][3] for idx in regions_to_merge]
			
			x_min = min(all_x)
			y_min = min(all_y)
			x_max = max(all_x_max)
			y_max = max(all_y_max)
			
			# Kreiranje novog spojenog regiona
			x3, y3 = x_min, y_min
			w3, h3 = x_max - x_min, y_max - y_min
			region3 = image_bin[y3:y3+h3, x3:x3+w3]
			
			connected_regions.append([resize_region(region3), (x3, y3, w3, h3)])

	# Filtriramo originalne regione - uklanjamo one koji su spojeni u veće regione
	filtered_regions = []
	for i, (region, bbox) in enumerate(regions_array):
		if i not in used_regions:
			filtered_regions.append([region, bbox])
    
    # Dodajemo spojene regione
	for connected_region, bbox in connected_regions:
		filtered_regions.append([connected_region, bbox])
	
	#
	#
	# ---------- KRAJ POBOLJŠANOG DELA ZA KVAČICE ----------
	#
	#
	
	#
	#
	# ---------- NOVO: BRISANJE MALIH ROI REGIONA ----------
	#
	#
	
	# Izračunavamo prosečnu visinu i širinu svih regiona
	heights = [bbox[3] for _, bbox in filtered_regions]
	widths = [bbox[2] for _, bbox in filtered_regions]
	
	if heights:  # Proveravamo da lista nije prazna
		avg_height = np.mean(heights)
		avg_width = np.mean(widths)
		
		# Definišemo pragove za male regione (interpunkcija)
		min_height_threshold = avg_height * 0.7  # Regioni manji od 70% prosečne visine
		min_width_threshold = avg_width * 0.5    # Regioni uži od 50% prosečne širine
		min_area_threshold = (avg_height * avg_width) * 0.5  # Regioni sa površinom manjom od 50% prosečne površine
		
		# Filtriramo male regione
		size_filtered_regions = []
		for region, (x, y, w, h) in filtered_regions:
			area = w * h
			
			# Zadržavamo region ako ne ispunjava uslove za brisanje
			# Ne dodajemo region ako je PREVIŠE mali po visini i širini i površini
			if not (h < min_height_threshold or w < min_width_threshold or area < min_area_threshold):
				size_filtered_regions.append([region, (x, y, w, h)])
		
		filtered_regions = size_filtered_regions
	
	#
	#
	# ---------- KRAJ DELA ZA BRISANJE MALIH ROI ----------
	#
	#

	regions_array = filtered_regions
	# Pripremamo podatke za NMS funkciju jer ona uzima podatke u obliku [x1, y1, x2, y2] (gornje levo teme i donje desno teme pravougaonika)
	# Dok mi radimo ovde samo sa (y1, x1) <- dovoljno je znati samo gornje levo teme jer znamo da su svi auti dimenzija 100x40 (width x height)
	boundingBoxes = np.zeros((len(regions_array), 4))
	for i in range(len(regions_array)):
		x, y, w, h = regions_array[i][1]
		boundingBoxes[i] = [x, y, x + w, y + h]

	# Primenjujemo NMS
	boundingBoxes = non_max_suppression_slow(boundingBoxes, 0.3)
	# Vracamo podatke u oblik [x, y, w, h]
	boundingBoxes = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boundingBoxes]

	# Pravimo skup od tih podataka, u skupu se nalaze samo x vrednosti koordinata
	bounding_box_set = set(box[0] for box in boundingBoxes)
	# Filtriramo regions_array kako bismo samo zadrzali elemente koje nam je NMS vratio
	filtered_regions = [
        item for item in regions_array 
        if item[1][0] in bounding_box_set
    ]

	# Iscrtavamo pravougaonike oko svakog od filtriranih regiona
	for _, (x,y,w,h) in filtered_regions:
		cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Sortiramo filtrirane regione
	filtered_regions = sorted(filtered_regions, key=lambda x: x[1][0])

	# Izdvajamo same regione
	sorted_regions = [region[0] for region in filtered_regions]

	# Izdvajamo centre regione u oblik (x_centar, y_centar)
	regions_centers = [] # lista centara samih karaktera
	regions_centers = [(region[1][0] + region[1][2] // 2, region[1][1] + region[1][3] // 2) for region in filtered_regions]
	
	# Izdvajamo [x, y, w, h] iz regiona
	sorted_rectangles = [region[1] for region in filtered_regions]
	return image_orig, sorted_regions, regions_centers, sorted_rectangles


def extract_text(image, model_nn, alphabet_labels):
    # Ovde ce biti rezultat
    test_result = ''
    # Ucitamo sliku
    test_color = image.copy()
    test_bin = image_bin(image_gray(test_color))

    # Oznacavamo ROI i prikazujemo sliku
    selected_roi_image, sorted_regions, regions_centers, sorted_rectangles = select_roi_test(test_color.copy(), invert_bin(test_bin))
    display_image(selected_roi_image)

    # Trazimo prosecne visine karaktera
    average_roi_height = 2.2 * np.mean(np.array([h for x, y, w, h in sorted_rectangles]))

    # Pravimo temp jer cemo iz njega u algoritmu brisati elemente 
    sorted_regions_temp = sorted_regions.copy()

    i = 0

    while len(regions_centers) > 0:
        # 1. Uzeti minimalnu y-vrednost centra karaktera u samom nizu centara i izdvojiti ga u poseban niz i obrisati iz trenutnog
        minv = min(regions_centers, key=lambda x: x[1])
        # Nabavljamo indeks minimalnog kako bismo znali sa kog indeksa brisemo u drugim nizovima
        minv_ind = regions_centers.index(minv)

        simillar_center = [regions_centers[minv_ind]]
        sorted_regions_row = [sorted_regions_temp[minv_ind]]
        sorted_rectangles_row = [sorted_rectangles[minv_ind]]

        # Brisanje elementa iz nizova
        regions_centers.remove(minv)
        del sorted_regions_temp[minv_ind]
        del sorted_rectangles[minv_ind]

        # 2. Proci kroz ostatak niza i proveriti da li postoje karakteri sa slicnom y-vrednost centra (y-centar +- (prosecna-visina-svih-karaktera / 2)) 
        # i dodati ih u posebne nizove i obrisati iz trenutnih nizova
        simillar_center += [el for el in regions_centers if el[1] >= minv[1] - int(average_roi_height) and el[1] <= minv[1] + int(average_roi_height)]

        for x in simillar_center:
            if x in regions_centers:
                ind = regions_centers.index(x)
                regions_centers.remove(x)
                sorted_regions_row += [sorted_regions_temp[ind]]
                sorted_rectangles_row += [sorted_rectangles[ind]]
                del sorted_regions_temp[ind]
                del sorted_rectangles[ind]

        # Korisceno da vidim da li sve radi kako sam zeleo :)
        #print(len(regions_centers), len(sorted_regions_temp), len(sorted_rectangles))
        #print(len(simillar_center), len(sorted_regions_row), len(sorted_rectangles_row))

        # 3. Sortirati poseban niz po x-vrednosti centara i sracunati udaljenosti izmedju svaka 2 karaktera kako bismo mogli uraditi KMeans nad ovim redom
        # Spajamo regione i centre u jedan niz
        sorted_regions_and_centers_and_rectangles = []
        for i in range(len(sorted_regions_row)):
            sorted_regions_and_centers_and_rectangles += [[sorted_regions_row[i], simillar_center[i], sorted_rectangles_row[i]]]

        # Sortiramo niz po x vrednosti i izvlacimo regione i centre u poseban niz
        # Ovaj korak je bio potreban jer nisam bio siguran u to da li ce elementi ocuvati sortiranost iz prethodnih koraka
        sorted_regions_and_centers_and_rectangles = sorted(sorted_regions_and_centers_and_rectangles, key=lambda x: x[1][0])
        sorted_regions_row = [region[0] for region in sorted_regions_and_centers_and_rectangles]
        sorted_centers_row = [region[1] for region in sorted_regions_and_centers_and_rectangles]
        sorted_rectangles_row = [region[2] for region in sorted_regions_and_centers_and_rectangles]

        sorted_regions_row = [invert_bin(region) for region in sorted_regions_row]

        # Racunamo distance i primenjujemo KMeans
        if len(sorted_regions_row) > 2:
            region_distances = []
            for index in range(0, len(sorted_rectangles_row) - 1):
                current = sorted_rectangles_row[index]
                next_rect = sorted_rectangles_row[index + 1]
                distance = next_rect[0] - (current[0] + current[2]) # x_next - (x_current + w_current)
                region_distances.append(distance)
    
            region_distances = np.array(region_distances).reshape(len(region_distances), 1)
            k_means = KMeans(n_clusters=2, n_init=10)
            k_means.fit(region_distances)

            # 4. Prikazemo rezultat samo tog jednog reda sa razmacima i dodamo '\n' na kraj rezultata
            test_inputs = prepare_for_cnn(sorted_regions_row)
            result = model_nn.predict(test_inputs)

            test_result += display_result_with_spaces(result, alphabet_labels, k_means)
            test_result += '\n'
        else:
            # 4. Prikazemo rezultat samo tog jednog reda sa razmacima i dodamo '\n' na kraj rezultata
            test_inputs = prepare_for_cnn(sorted_regions_row)
            result = model_nn.predict(test_inputs)

            test_result += display_result_with_spaces(result, alphabet_labels, None)
            test_result += '\n'

        # 5. Ponavljamo korake 1-4 sve dok ne dobijemo prazan red centara

    return test_result

def prepare_for_cnn(regions):
    ready_for_ann = []
    for region in regions:
        resized_region = resize_region(region)

        region_with_channel = np.reshape(resized_region, (64, 64, 1))  # Dodavanje dimenzije kanala
        ready_for_ann.append(region_with_channel)
        
    return np.array(ready_for_ann, dtype=np.float32)

def display_result_with_spaces(outputs, alphabet, k_means):
    if k_means is None:
        return alphabet[winner(outputs[0])]    
                                                          
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]    # odredjuje rastojanje izmedju reci (max)
                                                                                       # enumerate daje parove
    result = alphabet[winner(outputs[0])]
                                                                                                                        
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result

def winner(output):                                                                    # odredjuje pobednicki neuron, onaj  neuron
    return max(enumerate(output), key=lambda x: x[1])[0]   

def process_image(image_path, ann, alphabet, display_steps=False):     #dodala sam display_steps cisto da mogu da biram jel zelim sve ispisano ili ne
    """
    Procesira sliku: učitavanje, ispravljanje zakošenosti, rotacija, binarizacija i OCR.
    Vraća prepoznat tekst.
    """
    try:
        if display_steps:
            print(f"Učitavam sliku: {image_path}")
        original_image = load_image(image_path)

        if original_image is None:
            raise ValueError("Slika nije uspešno učitana. Proverite putanju i integritet fajla.")

        if display_steps:
            print(f"Slika '{image_path}' uspešno učitana. Dimenzije: {original_image.shape}")
            display_image(original_image, color=True)

            print("\nKorak 1: Ispravljanje zakošenosti")
        skew_corrected_image = correct_skewed_image(original_image.copy())
        if display_steps:
            display_image(skew_corrected_image, color=False)

            print("\nKorak 2: Rotiranje slike")
        rotated_image = rotate_image_to_normal(skew_corrected_image.copy(), image_gray(skew_corrected_image.copy()))
        if display_steps:
            display_image(rotated_image, color=True)

            print("\nKorak 3: Binarizacija slike")
        rotated_image_bin = image_bin(image_gray(rotated_image))
        if display_steps:
            display_image(rotated_image_bin, color=False)

        if display_steps:
            print("\nKorak 4: Prepoznavanje teksta")
        
        recognized_text = extract_text(rotated_image_bin.copy(), ann, alphabet)

        if display_steps:
            print("\nKorak 5: Konačni rezultat:")
            # print(recognized_text)

        return recognized_text
        

    except FileNotFoundError:
        print(f"Greška: Slika na putanji '{image_path}' nije pronađena.")
    except NameError as ne:
        print(f"Greška: Nedostaje definicija '{ne}'")
    except ValueError as ve:
        print(f"Greška tokom obrade slike: {ve}")
    
    return None  # Ako nešto krene po zlu



# --- KRAJ FUNKCIJA ---

# Putanje do modela
MODELS = {
    'combined': 'ocr_model.h5',
}

# Inicijalizacija i ucitavanje modela
models = {}
for name, path in MODELS.items():
    try:
        models[name] = load_model(path)
        print(f"Model '{name}' uspešno učitan.")
    except Exception as e:
        print(f"Greška prilikom učitavanja modela '{name}': {e}")
        models[name] = None

# Mape karaktera
character_maps = {
    'combined': [ 
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'Č', 'Ć', 'Đ', 'Ž', 'Š', 
            'č', 'ć', 'đ', 'ž', 'š',
            'А', 'Б', 'В', 'Г', 'Д', 'Ђ', 'Е', 'Ж', 'З', 'И', 
            'Ј', 'К', 'Л', 'Љ', 'М', 'Н', 'Њ', 'О', 'П', 'Р', 
            'С', 'Т', 'Ћ', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Џ', 'Ш',
            'а', 'б', 'в', 'г', 'д', 'ђ', 'е', 'ж', 'з', 'и', 
            'ј', 'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п','р', 
            'с', 'т', 'ћ', 'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
            ]
}

# Funkcija za odabir slike
def open_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if not filepath:
        return
    file_label.config(text=filepath)

# Funkcija za prepoznavanje teksta sa zadatim modelom
def recognize_and_display(model_name):
    filepath = file_label.cget("text")
    if not filepath or not os.path.exists(filepath):
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "Molimo odaberite ispravnu sliku.")
        return
    
    selected_model = models.get(model_name)
    if not selected_model:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Greška: Model '{model_name}' nije učitan.")
        return

    selected_char_map = character_maps.get(model_name)
    if not selected_char_map:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Greška: Mapa karaktera za '{model_name}' nije pronađena.")
        return

    try:
        recognized_text = process_image(filepath, selected_model, selected_char_map)

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, recognized_text)

    except Exception as e:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, f"Došlo je do greške: {e}")

# Kreiranje glavnog prozora
window = tk.Tk()
window.title("OCR Aplikacija")
window.geometry("600x450")

# Kreiranje elemenata korisničkog interfejsa
title_label = tk.Label(window, text="Prepoznavanje teksta sa slike", font=("Helvetica", 16))
title_label.pack(pady=10)

file_frame = tk.Frame(window)
file_frame.pack(pady=5)

select_button = tk.Button(file_frame, text="Odaberi sliku", command=open_file)
select_button.pack(side=tk.LEFT, padx=5)

file_label = tk.Label(file_frame, text="Nije odabrana nijedna slika.", width=40)
file_label.pack(side=tk.LEFT)

# Dugmad za prepoznavanje
button_frame = tk.Frame(window)
button_frame.pack(pady=10)

combined_button = tk.Button(button_frame, text="Pretvori u tekst (kombinovano)", command=lambda: recognize_and_display('combined'))
combined_button.pack(side=tk.LEFT, padx=5)

output_label = tk.Label(window, text="Prepoznati tekst:", font=("Helvetica", 12))
output_label.pack(pady=5)

output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=12)
output_text.pack(pady=5)

# Pokretanje glavne petlje Tkinter-a
window.mainloop()