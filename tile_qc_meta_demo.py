#!/usr/bin/env python

''' Analyzes raw image tiles using matcher data from metafile.
    Jay Borseth 2017.02.06

1. Extracts info from the matcher in the metafile on the bottom and left or right side of each tile,
   which creates matching score, distance from ideal, x_offset from ideal, y_offset from ideal, and focus images.

2. Interactively displays these images allowing switching between them with the keyboard:
   Select one of these first:
      r: show info by rows
      c: show info by columns

   f: focus
   s: std_dev of grayscale pixel values
   d: distance from ideal 
   x: x_offset from ideal 
   y: y_offset from ideal 
   q: qualtiy of match

   [0-9]: opacity of background montage image

3. Ctrl+Click the mouse to show the local area tiles in a zoomable viewer.
    t: toggle viewing of tile boundaries
    m: toggle viewing of tiles using matcher locations

   Shift+Click the mouse to show an individual tile at full resolution in Irfanview.

4. Saves all produced images to an output directory.

5. Uses a yaml "config_file" which must contain a "tile_qc" section to perform checks and creat a go/nogo decision on whether the 
   montage is of sufficient quality.  At present, the valid keys are:

    tile_qc:
        max_match_0: 0 # how many tiles with 0 matches are allowed?
        max_match_1: 0 # how many tiles with 1 matches are allowed?
        min_mean_std_dev: 5 # minimum average standard deviation of pixel values allowed?
        min_focus: 16 # minimum focus value allowed?  

    Results are written to "results_file".

'''

    
  

import argparse
import collections
import glob
import json
import logging
import multiprocessing
import os
import sys
import time
import math
from copy import deepcopy
from itertools import islice
from subprocess import Popen
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pylab
import pyqtgraph as pg
import yaml
from pyqtgraph.Qt import QtCore, QtGui

import cv2


class LruCache:
    ''' https://www.kunxi.org/blog/2014/05/lru-cache-in-python/ '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


def item_from_raster_pos(args_col_row, col, row):
    ''' returns a node given a col, row'''
    prop_bag = args_col_row[3]
    return prop_bag.raster_pos_lookup[str(col) + "_" + str(row)]


def create_raster_pos_dict(args_col_row):
    ''' create the look up dictionary for raster pos to nodes '''
    prop_bag = args_col_row[3]
    prop_bag.raster_pos_lookup = {}
    args = args_col_row[2]
    if not 'data' in args:
        args.meta_file, args.montage_file = get_meta_and_montage_files(
            args.directory)

        with open(args.meta_file) as data_file:
            json_data = json.load(data_file)
            args.metadata = json_data[0]['metadata']
            args.data = json_data[1]['data']

    for tile in args.data:
        rp = tile['img_meta']['raster_pos']
        col, row = rp
        prop_bag.raster_pos_lookup[str(col) + "_" + str(row)] = tile


def get_meta_and_montage_files(rootdir):
    '''get the names of the meta and montage files'''
    for name in glob.glob(os.path.join(rootdir, r"_meta*.*")):
        meta = name
    for name in glob.glob(os.path.join(rootdir, r"_montage*.*")):
        montage = name
    return (meta, montage)


def parse_metadata(args):
    ''' read in the metadata file and extract relevant info'''
    rootdir = args.directory
    try:
        args.meta_file, args.montage_file = get_meta_and_montage_files(rootdir)

        with open(args.meta_file) as data_file:
            json_data = json.load(data_file)
    except:
        raise Exception("Cannot find or parse metafile in: " + args.directory)

    metadata = args.metadata = json_data[0]['metadata']
    data = args.data = json_data[1]['data']

    args.overlap = metadata['overlap']
    args.overlap_beyond = args.overlap
    args.width = metadata["camera_info"]["width"]
    args.height = metadata["camera_info"]["height"]
    args.temca_id = metadata["temca_id"]
    try:
        args.n_edge_matches = metadata["n_edge_matches"]
    except:
        if metadata["roi_creation_time"] > "20190415": # about the time we switched to 10 matches per edge?
            args.n_edge_matches = 10
        else:
            args.n_edge_matches = 3  # phase II data

    # remove unwanted rows and columns
    all_rows = all_cols = 0
    row_start = col_start = 0
    row_end = col_end = 999999

    filtered_data = []
    for tile in data:
        rp = tile['img_meta']['raster_pos']
        col, row = rp
        all_rows = max(all_rows, row)
        all_cols = max(all_cols, col)
        if col >= col_start and row >= row_start and col <= col_end and row <= row_end:
            filtered_data.append(tile)
    args.data = filtered_data

    args.all_rows = all_rows
    args.all_cols = all_cols
    args.mask = np.zeros((all_rows+1, all_cols+1), dtype=np.uint8)

    args.raw_focus = np.empty((args.all_rows+1, args.all_cols+1))
    args.raw_focus.fill(0)

    args.raw_std_dev = np.empty((args.all_rows+1, args.all_cols+1))
    args.raw_std_dev.fill(0)

    # figure out the number of rows and columns and extract focus values
    trows = tcols = 0
    focus_min = 9999
    focus_max = -1
    for tile in args.data:
        rp = tile['img_meta']['raster_pos']
        col, row = rp
        args.mask[row][col] = 1
        #print row, col
        trows = max(trows, row)
        tcols = max(tcols, col)
        focus = tile["focus_score"]
        focus_min = min(focus, focus_min)
        focus_max = max(focus, focus_max)
        args.raw_focus[row][col] = focus
        # early metadata didn't have stddev
        if "std_dev" in tile:
            std_dev = tile["std_dev"]
            args.raw_std_dev[row][col] = std_dev


    # the mask
    args.mask_integer = args.mask.astype(int)

    # total number of rows and cols
    args.trows = trows
    args.tcols = tcols

    # allocate and fill column and row lists
    args.columns = {}  # {[{} for i in range(tcols+1)]}
    args.rows = {}  # [{} for i in range(trows+1)]

    # 6 planes, top_distance, top_x_offset, top_y_offset, side_distance, side_x_offset, side_y_offset,
    args.im_dist_to_ideal = np.empty((args.all_rows+1, args.all_cols+1, 6))
    args.im_dist_to_ideal.fill(0)
    # plane[0] = top, plane[1] = side
    args.im_match_quality = np.empty((args.all_rows+1, args.all_cols+1, 2))
    args.im_match_quality.fill(-1)

    mean_std_dev = 0.0
    for index, tile in enumerate(args.data):
        rp = tile['img_meta']['raster_pos']
        if 'std_dev' in tile:
            mean_std_dev += tile["std_dev"]
        col, row = rp
        if not col in args.columns:
            args.columns[col] = []
        args.columns[col].append(index)
        if not row in args.rows:
            args.rows[row] = []
        args.rows[row].append(index)
        args.mask[row][col] = 1
        if 'matcher' in tile:
            mt = tile['matcher'][0]  # matcher top
            ms = tile['matcher'][1]  # matcher side
            # 2 left, 3 top, 4 right
            if mt['position'] != 3:
                # swap them
                ms, mt = (mt, ms)
            args.im_dist_to_ideal[row][col][0] = mt['distance']
            args.im_dist_to_ideal[row][col][1] = mt['dX']
            args.im_dist_to_ideal[row][col][2] = mt['dY']
            args.im_dist_to_ideal[row][col][3] = ms['distance']
            args.im_dist_to_ideal[row][col][4] = ms['dX']
            args.im_dist_to_ideal[row][col][5] = ms['dY']

            # early files didn't have n_edge_matches, so try and figure it out
            if mt['match_quality'] > args.n_edge_matches:
                args.n_edge_matches = mt['match_quality']
            if ms['match_quality'] > args.n_edge_matches:
                args.n_edge_matches = ms['match_quality']

            args.im_match_quality[row][col][0] = mt['match_quality']
            args.im_match_quality[row][col][1] = ms['match_quality']

    args.mean_std_dev = mean_std_dev / len(args.data)

def init_visualization(args):
    ''' visualization levels, subsequent level include previous levels
    0 - None, just write output files
    1 - show resulting heatmaps
    '''
    if args.visualization >= 1:
        pass


def LUT_from_colormap(colormap):
    ''' return a LUT given a cmap'''
    NUM_COLORS = 256

    # https://matplotlib.org/examples/color/colormaps_reference.html
    cm = pylab.get_cmap(colormap)
    LUT = np.zeros((256, 1, 3), dtype='uint8')
    for i in range(NUM_COLORS):
        # color will now be an RGBA tuple
        color = np.array(cm(1.*i/NUM_COLORS))
        color = 255 * color
        LUT[i][0:3] = color[0:3]
    return LUT


class PropertyBag(object):
    '''holds properties for visualization of both rows and cols'''

    def __init__(self):
        self.click_row = 0
        self.mouse_row = 0
        self.click_col = 0
        self.mouse_col = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.annotation_height = 100
        self.opacity = 0.2
        self.loading_from_pickle_file = False
        self.displaying = 'd'




def append_to_json_file(_dict,path):
    ''' never manually edit the final characters in this file '''
    with open(path, 'ab+') as f:
        f.seek(0,2)                                #Go to the end of file    
        if f.tell() == 0 :                         #Check if file is empty
            f.write(json.dumps([_dict], indent=4, sort_keys=True).encode())  #If empty, write an array
        else :
            f.seek(-1,2)
            f.truncate()                           #Remove the last character, open the array
            f.write(' , '.encode())                #Write the separator
            f.write(json.dumps(_dict, indent=4, sort_keys=True).encode())    #Dump the dictionary
            f.write(']'.encode())                  #Close the array


def analyze_results(args_col_row):
    ''' Analyze the qc results, and optionally append to a file which contains the overall qc results.
        returns a dict containing info on failures and a boolean failed. '''
    args = args_col_row[2]

    output = {
        "metadata": args.metadata,
        "data": args.data,
        "tile_qc": {
            "failed": False
            }
        }
        
    if args.config_file:
        try:
            config_file_handle = open(args.config_file, "r")
        except Exception as e:
            raise Exception ("Could not open config_file" + args.config_file + str(e))

        try:
            cfg = yaml.load(config_file_handle, Loader=yaml.SafeLoader)["tile_qc"]
        except Exception as e:
            raise Exception ("loading file with tile_qc section: " + args.config_file + str(e))

        config_file_handle.close()

        # all of the QC verification checks
        if "min_matches_per_edge" in cfg:
            # this is the new post phase II data format with 10? matches per edge
            min_matches_per_edge = cfg["min_matches_per_edge"]
            # /2 below because im_match_quality is actually a pair of columnwise and rowwise [r,c,2]}
            bad_edges = ((args.im_match_quality < min_matches_per_edge) & (args.im_match_quality >= 0)).sum() / 2
            output["tile_qc"]["bad_edges"] = bad_edges
            if "allowed_match_failures" in cfg:
                allowed_match_failures = cfg["allowed_match_failures"]
                if bad_edges > allowed_match_failures:
                    output["tile_qc"]["failed_allowed_match_failures"] = "{} > {}".format(bad_edges, allowed_match_failures)
                    output["tile_qc"]["failed"] = True

        elif "max_match_0" in cfg:
            # this is the old phase II data format with only 3 matches per edge
            # how many edge matches of each type?
            match_0 = np.count_nonzero(args.im_match_quality == 0)
            match_1 = np.count_nonzero(args.im_match_quality == 1)

            # no matches
            output["tile_qc"]["match_0"]= match_0
            if "max_match_0" in cfg:
                if match_0 > cfg["max_match_0"]:
                    output["tile_qc"]["failed_match_0"] = "{} > {}".format(match_0, cfg["max_match_0"])
                    output["tile_qc"]["failed"] = True
            # only 1 match
            output["tile_qc"]["match_1"]= match_1
            if "max_match_1" in cfg:
                if match_1 > cfg["max_match_1"]:
                    output["tile_qc"]["failed_match_1"] =  "{} > {}".format(match_1, cfg["max_match_1"])
                    output["tile_qc"]["failed"] = True

        # mean of std_devs
        output["tile_qc"]["mean_std_dev"] = args.mean_std_dev
        if "min_mean_std_dev" in cfg:
            if args.mean_std_dev < cfg["min_mean_std_dev"]:
                output["tile_qc"]["failed_min_mean_std_dev"] =  "{} < {}".format(args.mean_std_dev, cfg["min_mean_std_dev"])
                output["tile_qc"]["failed"] = True
        # focus mean, std_dev
        focus_values = args.raw_focus[np.nonzero(args.raw_focus)]
        focus_std = np.std(focus_values)
        output["tile_qc"]["focus_mean"] = np.mean(focus_values)
        output["tile_qc"]["focus_std_dev"] = focus_std
        if "max_mean_focus_std_dev" in cfg:
            if focus_std < cfg["max_mean_focus_std_dev"]:
                output["tile_qc"]["failed_max_mean_focus_std_dev"] =  "{} < {}".format(focus_std, cfg["max_mean_focus_std_dev"])
                output["tile_qc"]["failed"] = True
        # aperture centroid shift
        actual_centroid = args.metadata["aperture_centroid"]
        if actual_centroid and "aperture_centroid" in cfg:
            theoretical_centroid = cfg["aperture_centroid"]
            allowed_shift = cfg["max_centroid_shift"]
            actual_shift = math.sqrt(((theoretical_centroid["x"] - actual_centroid["x"]) ** 2) + ((theoretical_centroid["y"] - actual_centroid["y"]) ** 2))
            output["tile_qc"]["centroid_shift"] = actual_shift
            if "max_centroid_shift" in cfg and (actual_shift > allowed_shift):
                output["tile_qc"]["failed_max_centroid_shift"] =  "{} > {}".format(actual_shift, allowed_shift)
                output["tile_qc"]["failed"] = True
        if args.results_file:
            append_to_json_file(output["tile_qc"], args.results_file)

    return output


def display_results(args_col_row):
    ''' show windows with image background and color overlays of distance to ideal and match quality.
    args_col_row is a list containing a separate copy of args for [args_columns, args_rows, args_active]
    Just toggle between the two complete data sets by setting args_col_row[2] to either of the first two elements. 
    Yes, it's a horrible but expeditious hack.'''
    prop_bag = args_col_row[3]
    for args in args_col_row[0:2]:
        if not args:
            continue
        args_col_row[2] = args

        directory = args.directory
        image_montage = cv2.imread(
            args.montage_file, flags=cv2.IMREAD_GRAYSCALE)
        if image_montage is None:
            raise Exception(
                "Could not load montage image.  The image tile directory path is wrong.")

        h, w = image_montage.shape

        image_montage = cv2.resize(
            image_montage, (int(w / prop_bag.scale_factor), int(h / prop_bag.scale_factor)), interpolation=cv2.INTER_AREA)
        prop_bag.scaled_montage_h, prop_bag.scaled_montage_w = image_montage.shape
        image_montage = cv2.equalizeHist(image_montage)
        args.image_montage = cv2.cvtColor(image_montage, cv2.COLOR_GRAY2BGR)

        # minmax, norm, uint8, colormap
        args.LUT_bwr = LUT_from_colormap('bwr')
        args.LUT_reds = LUT_from_colormap("Reds")
        args.LUT_jet = LUT_from_colormap('jet')

        # focus
        minv, maxv = cv2.minMaxLoc(args.raw_focus, args.mask)[0:2]  # focus
        args.raw_focus[args.mask] = minv
        if minv == maxv:
            maxv = minv+1
        scaled = (args.raw_focus - minv) / (maxv - minv) * 255
        args.im_focus_normalized_8 = np.uint8(scaled)
        args.im_focus_colormap = cv2.applyColorMap(
            args.im_focus_normalized_8, cv2.COLORMAP_JET)
        args.focus_minmax = (minv, maxv)

        max_possible_offset = int(
            (args.overlap + args.overlap_beyond) * args.width)

        # std_dev
        minv, maxv = cv2.minMaxLoc(args.raw_std_dev, args.mask)[0:2]
        args.raw_std_dev[args.mask] = minv
        if minv == maxv:
            maxv = minv+1
        scaled = (args.raw_std_dev - minv) / (maxv - minv) * 255
        args.im_std_dev_normalized_8 = np.uint8(scaled)
        args.im_std_dev_colormap = cv2.applyColorMap(
            args.im_std_dev_normalized_8, cv2.COLORMAP_JET)
        args.std_dev_minmax = (minv, maxv)

        # dist_to_ideal
        minv, maxv = cv2.minMaxLoc(args.im_dist_to_ideal[:, :, 0], args.mask)[
            0:2]  # distance
        #args.im_dist_to_ideal[:,:,0][args.mask] = minv
        scaled = (args.im_dist_to_ideal[:, :, 0]) / max_possible_offset * 255
        args.im_dist_to_ideal_normalized_8 = np.uint8(scaled)
        rgb = cv2.cvtColor(
            args.im_dist_to_ideal_normalized_8, cv2.COLOR_GRAY2BGR)
        args.im_dist_to_ideal_colormap = cv2.LUT(rgb, args.LUT_reds)
        args.im_dist_to_ideal_colormap = cv2.cvtColor(
            args.im_dist_to_ideal_colormap, cv2.COLOR_BGR2RGB)  # FSK?
        args.distance_minmax = (minv, maxv)

        minv, maxv = cv2.minMaxLoc(args.im_dist_to_ideal[:, :, 1], args.mask)[
            0:2]  # x_offset
        #args.im_dist_to_ideal[:,:,1][args.mask] = minv
        scaled = args.im_dist_to_ideal[:, :, 1] / max_possible_offset * 255
        scaled += 128
        args.im_x_offset_normalized_8 = np.uint8(scaled)
        rgb = cv2.cvtColor(args.im_x_offset_normalized_8, cv2.COLOR_GRAY2BGR)
        args.im_x_offset_colormap = cv2.LUT(rgb, args.LUT_bwr)
        args.x_offset_minmax = (minv, maxv)

        minv, maxv = cv2.minMaxLoc(args.im_dist_to_ideal[:, :, 2], args.mask)[
            0:2]  # y_offset
        #args.im_dist_to_ideal[:,:,2][args.mask] = minv
        scaled = args.im_dist_to_ideal[:, :, 2] / max_possible_offset * 255
        scaled += 128
        args.im_y_offset_normalized_8 = np.uint8(scaled)
        rgb = cv2.cvtColor(args.im_y_offset_normalized_8, cv2.COLOR_GRAY2BGR)
        args.im_y_offset_colormap = cv2.LUT(rgb, args.LUT_bwr)
        args.y_offset_minmax = (minv, maxv)

        h, w = args.im_match_quality.shape
        args.im_match_quality_colormap = np.zeros((h, w, 3), dtype="uint8")
        args.im_match_quality_colormap.fill(-1)

        if args.n_edge_matches <= 3:
            args.im_match_quality_colormap[np.where(args.im_match_quality==[0])] = [0, 0, 255]  # no match
            args.im_match_quality_colormap[np.where(args.im_match_quality==[-1])] = [255, 255, 255]  # invalid (edges of image)
            args.im_match_quality_colormap[np.where(args.im_match_quality==[2])] = [0, 255, 255]  # 2 matches
            args.im_match_quality_colormap[np.where(args.im_match_quality==[3])] = [40, 40, 40]  # 3 matches

            args.quality_minmax = (-1, 3)

        else:
            args.im_match_quality_colormap[np.where((args.im_match_quality!=[-1]) & (args.im_match_quality<=[3]))] = [0, 0, 255]  # no match
            #                    0  1  2  3  4   5   6   7   8   9  10 
            blue_intensity_ar = [0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 60]
            for j in range(4, 8):
                bi = blue_intensity_ar[j]
                args.im_match_quality_colormap[np.where(args.im_match_quality==[j])] = [bi, 255, 255] #
            args.quality_minmax = (-1, 10)

        # Resize images
        args.im_dist_to_ideal_final = cv2.resize(
            args.im_dist_to_ideal_colormap, (prop_bag.scaled_montage_w,
                                             prop_bag.scaled_montage_h),
            interpolation=cv2.INTER_AREA)

        args.im_x_offset_final = cv2.resize(
            args.im_x_offset_colormap, (prop_bag.scaled_montage_w,
                                        prop_bag.scaled_montage_h),
            interpolation=cv2.INTER_AREA)

        args.im_y_offset_final = cv2.resize(
            args.im_y_offset_colormap, (prop_bag.scaled_montage_w,
                                        prop_bag.scaled_montage_h),
            interpolation=cv2.INTER_AREA)

        args.im_match_quality_final = cv2.resize(
            args.im_match_quality_colormap, (prop_bag.scaled_montage_w,
                                             prop_bag.scaled_montage_h),
            interpolation=cv2.INTER_AREA)

        args.im_focus_final = cv2.resize(
            args.im_focus_colormap, (prop_bag.scaled_montage_w,
                                     prop_bag.scaled_montage_h),
            interpolation=cv2.INTER_AREA)

        args.im_std_dev_final = cv2.resize(
            args.im_std_dev_colormap, (prop_bag.scaled_montage_w,
                                     prop_bag.scaled_montage_h),
            interpolation=cv2.INTER_AREA)

    # im_out is a composite used for writing to file and displaying the overall results
    prop_bag.im_out = np.zeros((prop_bag.scaled_montage_h + prop_bag.annotation_height,
                                prop_bag.scaled_montage_w, 3), np.uint8)  # composite image with annotations

    args = args_col_row[2]

    prop_bag.output = analyze_results(args_col_row) # write the results to file

    if args.image_directory or args.copy_to_metadata_dir:
        prop_bag.output["qc_image_paths"] = save_images(args_col_row)
    else:
        prop_bag.output["qc_image_paths"] = {} # relative paths to the 10 output images

    if args.visualization >= 1:
        create_raster_pos_dict(args_col_row)
        s = args.metadata["session_id"] + "_" + str(args.metadata["grid"]) + "_" + str(args.metadata["roi_index"]) \
            + '    Keys: "R"ows, "C"ols, "F"ocus, "S"td_dev, D"istance, "X", "Y", (0-9) opacity '
        prop_bag.named_window = s
        cv2.namedWindow(prop_bag.named_window, cv2.WINDOW_NORMAL)
        win_size = prop_bag.im_out.shape
        cv2.resizeWindow(prop_bag.named_window, win_size[1], win_size[0])
        cv2.setMouseCallback(prop_bag.named_window, on_mouse_event, args_col_row)
        while update_display_until_esc(args_col_row):
            pass
        cv2.destroyWindow(prop_bag.named_window)


def save_images(args_col_row):
    ''' create all images and save them if an image directory was specified,
    or if we're writing images back to the metadata directory.
    Returns a dict of relative paths to all 10 of the QC images.
    ["focus_row"] = _QC... 
    '''
    args = args_col_row[2]
    prop_bag = args_col_row[3]
    copy_to_directory = getattr(args, "image_directory")
    copy_to_metadata_dir = getattr(args, "copy_to_metadata_dir", True)
    qc_image_paths = {}
    if copy_to_directory or copy_to_metadata_dir:
        try:
            if not os.path.exists(args.image_directory):
                os.makedirs(args.image_directory)
        except:
            raise Exception(
                'Unable to make image directory: ' + args.image_directory)
        for args in args_col_row[0:2]:
            if args:
                for im_type in [('f', 'focus'), ('s', 'std_dev'), ('d', 'distance'), ('x', 'x_offset'), ('y', 'y_offset'), ('q', 'quality')]:
                    prop_bag.displaying = im_type[0]
                    create_image_with_annotations(args, prop_bag)
                    # files from graph matcher have '_m' appended
                    s = "_QC_" + args.metadata["roi_creation_time"] + "_" + args.metadata["session_id"] + "_" + str(args.metadata["grid"]) + "_" + str(args.metadata["roi_index"]) \
                        + '_' + im_type[1] + "_" + args.orientation + "_m" + ".jpg"
                    qc_image_paths[im_type[1]+"_"+ args.orientation] = s
                    if copy_to_directory:
                        ipath = os.path.join(args.image_directory, s)
                        cv2.imwrite(ipath, prop_bag.im_out)
                    if copy_to_metadata_dir:
                        ipath = os.path.join(args.directory, s)
                        ok = cv2.imwrite(ipath, prop_bag.im_out)
    return qc_image_paths                


def create_image_with_annotations(args, prop_bag):
    ''' create an output image with annotations
    Sets prop_bag.im_out with annotated composite image'''

    dstr = ""

    if prop_bag.displaying == 'f':  # focus
        im_view = args.im_focus_final
        minmax = args.focus_minmax
        dstr = 'focus: {:4.2f}'.format(
            args.raw_focus[prop_bag.mouse_row][prop_bag.mouse_col])
        istr = 'focus score'
    elif prop_bag.displaying == 's':  # std_dev
        im_view = args.im_std_dev_final
        minmax = args.std_dev_minmax
        dstr = 'std_dev: {:4.2f}'.format(
            args.raw_std_dev[prop_bag.mouse_row][prop_bag.mouse_col])
        istr = 'std_dev'
    elif prop_bag.displaying == 'd':  # distance
        im_view = args.im_dist_to_ideal_final
        minmax = args.distance_minmax
        dstr = 'dist: {:4.0f}'.format(
            args.im_dist_to_ideal[prop_bag.mouse_row][prop_bag.mouse_col][0])
        istr = 'distance from ideal'
    elif prop_bag.displaying == 'x':  # x distance
        im_view = args.im_x_offset_final
        minmax = args.x_offset_minmax
        dstr = 'x: {:4.0f}'.format(
            args.im_dist_to_ideal[prop_bag.mouse_row][prop_bag.mouse_col][1])
        istr = 'x_offset from ideal'
    elif prop_bag.displaying == 'y':  # y distance
        im_view = args.im_y_offset_final
        minmax = args.y_offset_minmax
        dstr = 'y: {:4.0f}'.format(
            args.im_dist_to_ideal[prop_bag.mouse_row][prop_bag.mouse_col][2])
        istr = 'y_offset from ideal'
    elif prop_bag.displaying == 'q':  # quality
        im_view = args.im_match_quality_final
        minmax = args.quality_minmax
        dstr = 'Q: {:4.0f}'.format(
            args.im_match_quality[prop_bag.mouse_row][prop_bag.mouse_col])
        istr = 'match quality'

    dst = cv2.addWeighted(args.image_montage, prop_bag.opacity,
                          im_view, 1 - prop_bag.opacity, 0)

    prop_bag.im_out[prop_bag.scaled_montage_h:, :, :] = 0
    prop_bag.im_out[0:prop_bag.scaled_montage_h, :,
                    :] = dst  # copy to top portion of im_out

    spacing = 24
    text_start = prop_bag.scaled_montage_h + 18
    # temca_id, orientation, typ
    s = "matcher: {}:    {} {}     min: {}, max: {},".format(
        args.temca_id, args.orientation, istr, int(minmax[0]), int(minmax[1]))
    cv2.putText(prop_bag.im_out, s, (2, text_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    # minmax + scale
    s = "overlap: {:.2f}, overlap_beyond: {:.2f}".format(
        args.overlap, args.overlap_beyond)
    cv2.putText(prop_bag.im_out, s, (2, text_start + spacing),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    # draw the colorbar legend
    # cbl = 300
    # prop_bag.im_out[text_start + spacing, cbl: cbl+256, :] = args.LUT_jet
    cv2.putText(prop_bag.im_out, "session_id: " + args.metadata["session_id"] + "    "  + args.metadata["roi_creation_time"], (
        2, text_start + 2 * spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(prop_bag.im_out, "directory:  " + args.directory, (2, text_start +
                                                                   3 * spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(prop_bag.im_out, dstr, (prop_bag.mouse_x + 4, prop_bag.mouse_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cr = "c: " + str(prop_bag.mouse_col) + " r: " + str(prop_bag.mouse_row)
    cv2.putText(prop_bag.im_out, cr, (prop_bag.mouse_x + 4, prop_bag.mouse_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def update_display_until_esc(args_col_row):
    ''' redraw composite image, return False if ESC is pressed '''
    args = args_col_row[2]
    prop_bag = args_col_row[3]
    create_image_with_annotations(args, prop_bag)
    cv2.imshow(prop_bag.named_window, prop_bag.im_out)

    if prop_bag.quit:
        return False

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        prop_bag.quit = True
        return False
    if key == -1:
        return True

    if key > 255:
        return True
    chrkey = chr(key)
    if chrkey == 'c' or chrkey == 'C':  # columns
        if args_col_row[0]:
            args_col_row[2] = args_col_row[0]
    elif chrkey == 'r' or chrkey == 'R':  # rows
        if args_col_row[1]:
            args_col_row[2] = args_col_row[1]
    if chrkey == 'x' or chrkey == 'X':  # xOffset
        prop_bag.displaying = 'x'
    elif chrkey == 'y' or chrkey == 'Y':  # yOffset
        prop_bag.displaying = 'y'
    elif chrkey == 'd' or chrkey == 'D':  # distance
        prop_bag.displaying = 'd'
    elif chrkey == 'f' or chrkey == 'F':  # focus
        prop_bag.displaying = 'f'
    elif chrkey == 's' or chrkey == 'S':  # std_dev
        prop_bag.displaying = 's'
    elif chrkey == 'q' or chrkey == 'Q':  # quality
        prop_bag.displaying = 'q'
    elif chrkey >= '0' and chrkey <= '9':  # opacity (0.1 - 0.9)
        prop_bag.opacity = int(chrkey) * 0.1
    elif chrkey == 'm' or chrkey == 'M':  # matcher is a toggle
        prop_bag.place_using_matcher = not prop_bag.place_using_matcher
    elif chrkey == 't' or chrkey == 'T':  # matcher is a toggle
        prop_bag.show_images = not prop_bag.show_images
    return True

def item_or_none(args, c, r):
    ''' try and get an item, return None if it doesn't exist'''
    try:
        item = item_from_raster_pos(args, c, r)
        return item
    except KeyError as e:
        return None

class LocalGraphicsWindow(pg.GraphicsLayoutWidget):
    """
    Convenience subclass of :class:`GraphicsLayoutWidget 
    <pyqtgraph.GraphicsLayoutWidget>`. This class is intended for use from 
    the interactive python prompt.
    """
    def __init__(self, prop_bag, title=None, size=(800,800), **kargs):
        self.prop_bag = prop_bag
        pg.mkQApp()
        pg.GraphicsLayoutWidget.__init__(self, **kargs)
        self.resize(*size)
        if title is not None:
            self.setWindowTitle(title)
        self.show()

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.prop_bag.quit = True
        if key >= QtCore.Qt.Key_1 and key <= QtCore.Qt.Key_9:
            self.prop_bag.redraw = True
            self.prop_bag.tile_neighbors = key - QtCore.Qt.Key_1 + 1
        if key == QtCore.Qt.Key_M:
            self.prop_bag.redraw = True
            self.prop_bag.place_using_matcher = not self.prop_bag.place_using_matcher
        if key == QtCore.Qt.Key_T:
            self.prop_bag.redraw = True
            self.prop_bag.show_images = not self.prop_bag.show_images



def display_neighborhood(args_col_row):
    ''' show a zoomable QTgraph window with the neighboring tiles '''
    args = args_col_row[2]
    prop_bag = args_col_row[3]
    last_row = None
    last_col = None
    neighbors = None
    place_using_matcher = None
    img_cache = LruCache (50)

    if getattr(prop_bag, 'qtwin', None) is None:

        # set up the QTgraph window
        prop_bag.qtwin = LocalGraphicsWindow(prop_bag)
        s = args.metadata["session_id"] + "_" + str(args.metadata["grid"]) + "_" + str(args.metadata["roi_index"]) \
            + '    Keys: (1-9) neighbors, T (show tiles), M (use matcher)'
        prop_bag.qtwin.setWindowTitle(s)
        prop_bag.qtplt = prop_bag.qtwin.addPlot()
        # make sure the pyqtgraph window stays square
        prop_bag.qtplt.setAspectLocked(lock=True, ratio=1)
        prop_bag.qtplt.invertY(True)
        pg.QtGui.QApplication.processEvents()  # process the pyqt init events

    pen3 = pg.mkPen((100, 200, 100), width = 2) # 3 matches
    pen2 = pg.mkPen((200, 200, 0), width = 2) # 2 matches
    pen0 = pg.mkPen((255, 0, 0), width = 3)

    def things_have_changed():
        '''abort and restart if center point or neighbor count has changed.'''
        return ((prop_bag.click_col != last_col) or (prop_bag.click_row != last_row) or
            (neighbors != prop_bag.tile_neighbors) or (place_using_matcher != prop_bag.place_using_matcher))

    while not prop_bag.quit:

        cr = prop_bag.click_row
        cc = prop_bag.click_col

        if not prop_bag.redraw and not things_have_changed():
            pg.QtGui.QApplication.processEvents()  # process the pyqt events
            time.sleep(.01)
            continue

        last_row = cr
        last_col = cc
        neighbors = prop_bag.tile_neighbors
        place_using_matcher = prop_bag.place_using_matcher
        prop_bag.redraw = False

        prop_bag.qtplt.clear()  # clear the images from the plot - prevents memory leak

        # clear out old data, extending one beyond the size of the drawing area
        for nr in range(-(neighbors+1), neighbors+2):  # for the row neighbors
            # for the column neighbors
            for nc in range(-(neighbors+1), neighbors+2):
                item = item_or_none(args_col_row, cc + nc, cr + nr)
                if item is None:
                    continue
                item['pos'] = None

        # position the tiles using matcher results
        if place_using_matcher:
            olap = 1-args.overlap

            # use the highest quality matches first

            for match_quality in range(args.n_edge_matches, -1, -1): # [3, 2, 0]:
                for nr in list(range(0, neighbors+1)) + list(reversed(list(range(-neighbors, 1)))):
                    for nc in list(range(0, neighbors+1)) + list(reversed(list(range(-neighbors, 1)))):

                        if things_have_changed():
                            continue

                        match_q = 0
                        # middle then down, middle then up by row
                        zc = cc + nc # z is location this iteration
                        zr = cr + nr

                        #print match_quality, zc, zr

                        item = item_or_none(args_col_row, zc, zr)
                        if item is None:
                            continue

                        # nominal position
                        dx, dy = (int(nc*args.width * olap), int(nr*args.height*olap))

                        # figure out where this one goes...
                        if item['pos'] is not None:
                            # its already been placed
                            continue

                        if 'matcher' in item:
                            mt = item['matcher'][0]  # matcher top
                            ms = item['matcher'][1]  # matcher side

                            # set the upper left corner of the center tile at (0, 0)
                            if nc == 0 and nr == 0:
                                item['pos'] = np.array([0, 0])
                                match_q = min(ms['match_quality'], mt['match_quality'])
                            else:
                                pass

                            # try side matches first

                            # left match
                            if ms['position'] == 2:
                                tile = item_or_none(args_col_row, zc - 1, zr) # get left tile
                                if tile is None:
                                    continue
                                if tile['pos'] is not None and ms['match_quality'] == match_quality:
                                    item['pos'] = tile['pos'] + np.array([args.width * olap + ms['dX'], ms['dY']])
                                    match_q = ms['match_quality']
                                else:
                                    # now check the right side of this guy
                                    tile = item_or_none(args_col_row, zc + 1, zr) # get left tile
                                    if tile is None:
                                        continue
                                    if tile['pos'] is not None:
                                        ts = tile['matcher'][1]  # matcher side
                                        if ts['match_quality'] == match_quality:
                                            item['pos'] = tile['pos'] + np.array([-args.width * olap - ts['dX'], -ts['dY']])
                                            match_q = ts['match_quality']

                            # right match
                            elif ms['position'] == 4:
                                tile = item_or_none(args_col_row, zc + 1, zr) # get right tile
                                if tile is None:
                                    continue
                                if  tile['pos'] is not None and ms['match_quality'] == match_quality:
                                    item['pos'] = tile['pos'] + np.array([-args.width * olap + ms['dX'], ms['dY']])
                                    match_q = ms['match_quality']
                                else:
                                    # now check the left side of this guy
                                    tile = item_or_none(args_col_row, zc - 1, zr) # get left tile
                                    if tile is None:
                                        continue
                                    if tile['pos'] is not None:
                                        ts = tile['matcher'][1]  # matcher side
                                        if ts['match_quality'] == match_quality:
                                            item['pos'] = tile['pos'] + np.array([args.width * olap - ts['dX'], -ts['dY']])
                                            match_q = ts['match_quality']

                            # top match
                            if item['pos'] is None and mt['position'] == 3:
                                tile = item_or_none(args_col_row, zc, zr-1) # get tile below
                                if tile is None:
                                    continue
                                if tile['pos'] is not None and mt['match_quality'] == match_quality:
                                    item['pos'] = tile['pos'] + np.array([-mt['dX'], args.height * olap - mt['dY']])
                                    match_q = mt['match_quality']
                                else:
                                    # now check the tile above
                                    tile = item_or_none(args_col_row, zc, zr+1) # get tile above
                                    if tile is None:
                                        continue
                                    if tile['pos'] is not None:
                                        tt = tile['matcher'][0]  # matcher top
                                        if tt['match_quality'] == match_quality:
                                            item['pos'] = tile['pos'] + np.array([+ tt['dX'], -args.height * olap +tt['dY']])
                                            match_q = tt['match_quality']

                        # Boo!  There is no match. Set the position to the default
                        if item['pos'] is None and (match_quality == 0):
                            item['pos'] = np.array([dx, dy])

                        if item['pos'] is not None:
                            # its been placed!
                            name = os.path.join(args.directory, item["img_path"])
                            nx = int(item['pos'][0])
                            ny = int(item['pos'][1])

                            if prop_bag.show_images:
                                # print match_quality, dx, dy, nx, ny, name
                                img = img_cache.get(name)
                                if img:
                                    print ('cached: ' + name)
                                else:
                                    frame = cv2.imread(name, flags=cv2.IMREAD_GRAYSCALE)
                                    if frame is None:
                                        print ('Missing: ' + name)
                                    img = pg.ImageItem()
                                    img.setOpts(axisOrder='row-major')
                                    img.setImage(frame)
                                    img.setLevels([50, 200])
                                    #img.opacity = 0.5
                                    #img.autoLevels = True
                                    #img.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
                                    img_cache.set(name, img)
                                img.resetTransform()
                                img.translate(nx, ny)
                                prop_bag.qtplt.addItem(img)
                            if args.n_edge_matches == 3:
                                if match_q == 3:
                                    pen = pen3
                                elif match_q == 2:
                                    pen = pen2
                                elif match_q == 0:
                                    pen = pen0
                            else:
                                if match_q > 3:
                                    pen = pen3
                                else:
                                    pen = pen0
                            
                            rect = QtGui.QGraphicsRectItem(QtCore.QRectF(nx, ny, args.width, args.height))
                            rect.setZValue(1)
                            if match_q != -1:
                                rect.setPen(pen)
                            prop_bag.qtplt.addItem(rect)

                        pg.QtGui.QApplication.processEvents()  # process the pyqt events

        # place in fixed positions
        else:
            for nr in range(-neighbors, neighbors+1):  # for the row neighbors
                # for the column neighbors
                for nc in range(-neighbors, neighbors+1):
                    if things_have_changed():
                        continue
                    item = item_or_none(args_col_row, cc + nc, cr + nr)
                    if item is None:
                        continue

                    name = os.path.join(args.directory, item["img_path"])
                    print (name)
                    if name:
                        dx, dy = (nc*(args.width + 40), nr*(args.height + 40))

                        if prop_bag.show_images:
                            img = img_cache.get(name)
                            if img:
                                print ('cached: ' + name)
                            else:
                                frame = cv2.imread(name, flags=cv2.IMREAD_GRAYSCALE)
                                if frame is None:
                                    print ('Missing: ' + name)
                                img = pg.ImageItem()
                                img.setOpts(axisOrder='row-major')
                                img.setImage(frame)
                                img.setLevels([50, 200])
                                img_cache.set(name, img)
                            img.resetTransform()
                            img.translate(dx, dy)
                            prop_bag.qtplt.addItem(img)

                        rect = QtGui.QGraphicsRectItem(QtCore.QRectF(dx, dy, args.width, args.height))
                        if not 'matcher' in item:
                            pen = pen3
                        else:
                            mt = item['matcher'][0]  # matcher top
                            ms = item['matcher'][1]  # matcher side
                            if mt['match_quality'] == -1:
                                match_q = ms['match_quality']
                            elif ms['match_quality'] == -1:
                                match_q = mt['match_quality']
                            else:
                                match_q = min(ms['match_quality'], mt['match_quality'])
                            if match_q >= 3:
                                pen = pen3
                            elif match_q == 2:
                                pen = pen2
                            elif match_q == 0:
                                pen = pen0
                        rect.setPen(pen)
                        prop_bag.qtplt.addItem(rect)

                        pg.QtGui.QApplication.processEvents()  # process the pyqt events


display_thread = None

def on_mouse_event(event, x, y, flags, args_col_row):
    ''' process mouse events '''
    global display_thread

    args = args_col_row[2]
    prop_bag = args_col_row[3]

    prop_bag.mouse_x = x
    prop_bag.mouse_y = y
    prop_bag.mouse_row = min(args.all_rows, int(
        y / float(prop_bag.scaled_montage_h) * (args.all_rows + 1)))
    # prevent lookups in the annotation area
    prop_bag.mouse_col = min(args.all_cols, int(
        x / float(prop_bag.scaled_montage_w) * (args.all_cols + 1)))

    if event == cv2.EVENT_LBUTTONDOWN:

        if (flags & cv2.EVENT_FLAG_CTRLKEY) or (flags & cv2.EVENT_FLAG_SHIFTKEY):

            prop_bag.click_row = min(args.all_rows, int(
                y / float(prop_bag.scaled_montage_h) * (args.all_rows + 1)))
            # prevent lookups in the annotation area
            prop_bag.click_col = min(args.all_cols, int(
                x / float(prop_bag.scaled_montage_w) * (args.all_cols + 1)))

        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            name = None
            for name in glob.glob(
                    os.path.join(args.directory, r"*_{}_{}.tif").format(prop_bag.click_col,
                                                                        prop_bag.click_row)):
                print (name)
            if name:
                try:
                    # 32 bit version
                    Popen([r"C:\Program Files (x86)\IrfanView\i_view32.exe", name])
                except:
                    # 64bit version
                    Popen([r"C:\Program Files\IrfanView\i_view64.exe", name])

        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            if display_thread is None:
                display_thread = Thread(target=display_neighborhood, args=(args_col_row,))
                display_thread.start()

def process(args):
    ''' the main thing.'''
    tstart = time.time()
    prop_bag = PropertyBag()  # holds common settings for both rows and columns
    prop_bag.tile_neighbors = getattr(args, "tile_neighbors", 2)
    prop_bag.scale_factor = getattr(args, "scale_factor", 8)
    prop_bag.place_using_matcher = getattr(args, "place_using_matcher", True) # use matcher to position tiles
    prop_bag.show_images = getattr(args, "show_images", True) # show the images (else just boundaries)
    prop_bag.redraw = False # Triggers redrawing
    prop_bag.quit = False
    prop_bag.output = None # json obj containing results of QC operation including 'failed' flag

    parse_metadata(args)

    print ('rows: ', args.all_rows, ', cols: ', args.all_cols)
    init_visualization(args)

    args.all_results = []
    args_col = deepcopy(args)
    args_col.orientation = 'col'
    args_col.im_dist_to_ideal = args_col.im_dist_to_ideal[:, :, :3]
    args_col.im_match_quality = args_col.im_match_quality[:, :, 0]
    args_row = deepcopy(args)
    args_row.orientation = 'row'
    args_row.im_dist_to_ideal = args_row.im_dist_to_ideal[:, :, 3:]
    args_row.im_match_quality = args_row.im_match_quality[:, :, 1]

    if True:
        display_thread = Thread(target=display_results, args=(
            [args_col, args_row, None, prop_bag],))
        display_thread.start()
        display_thread.join()

    return prop_bag.output


def main(args):
    parent_parser = argparse.ArgumentParser(description='Raw tile analyzer, matches from metadata.',
                                            epilog="Visualization KEYS:  q: quality_of_match, d: distance_from_ideal (pix), x: x_offset, y: y_offset, f: focus, [0-9]: opacity," +
                                            " MOUSE: Ctrl+click: open image with surrounding montage, Shift+click: open in Irfanview")

    parent_parser.add_argument('directory', help='the directory to process.',
                               metavar="", nargs='?', default=r"H:/data/005320/0")
    parent_parser.add_argument('-c', '--config_file', type=str, default=None, metavar="",
                               help='yaml file containing qc limits')
    parent_parser.add_argument('-r', '--results_file', type=str, default=None, metavar="",
                               help='text file where analysis results will be appended')
    parent_parser.add_argument('-v', '--visualization', type=int, default=1, metavar="",
                               help='visualize the results (0): none - only write output images, (1): show results in interactive window')
    parent_parser.add_argument('-i', '--image_directory', type=str, default=r"c:\images",
                               metavar="", help='Directory in which to save analysis images')
    parent_parser.add_argument('-s', '--scale_factor', type=int, default=8,
                               metavar="", help='output images are scaled by this factor, try 8 or 16')
    parent_parser.add_argument('-t', '--tile_neighbors', type=int, default=2,
                               metavar="", help='number of neighboring tiles to display on ctrl+click')
    parent_parser.add_argument('-w', '--show_images', type=bool, default=True,
                               metavar="", help='disable showing images to speed up showing just boundary rectangles')
    parent_parser.add_argument('-m', '--copy_to_metadata_dir', type=bool, default=True,
                               metavar="", help='also copy image files to metadata directory.')
    args = parent_parser.parse_args(args)

    process(args)


if __name__ == "__main__":
    main(sys.argv[1:])
