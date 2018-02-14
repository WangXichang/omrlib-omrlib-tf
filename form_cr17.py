# *_* utf-8 *_*

import omrlib as ol
import glob
# import numpy as np


def omr_read(card_form):
    return ol.omr_read_batch(card_form)


def form_17cr_A():
    omr_location = [r"d:\pythontest\omrphototest\A\omr\A86261\*",
                    r"d:\pythontest\omrphototest\A\omr\A86262\*",
                    r"d:\pythontest\omrphototest\A\omr\A86263\*",
                    r"d:\pythontest\omrphototest\A\omr\A86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')
    group = {
        j+5*h:[(j, 3+8*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    }
    group.update({
        j+20+5*h:[(j + 5, 3+8*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 11,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 90,
            'y_end': -1
        }
    }
    return card_form


def form_17cr_B():

    omr_location = [
                    r"d:\pythontest\omrphototest\B\omr\B86261\*",
                    r"d:\pythontest\omrphototest\B\omr\B86262\*",
                    r"d:\pythontest\omrphototest\B\omr\B86263\*",
                    r"d:\pythontest\omrphototest\B\omr\B86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+6*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 5)
    }
    group.update({
        j+25+5*h:[(j + 5, 3+6*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 2)
    })
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 11,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 90,
            'y_end': -1
        }
    }
    return card_form


def form_17cr_C():

    omr_location = [
                    r"d:\pythontest\omrphototest\c\omr\C86261\*",
                    r"d:\pythontest\omrphototest\c\omr\C86262\*",
                    r"d:\pythontest\omrphototest\c\omr\C86263\*",
                    r"d:\pythontest\omrphototest\c\omr\C86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+8*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    }

    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 90,
            'y_end': -1
        }
    }
    return card_form


def form_17cr_D():

    omr_location = [
                    r"d:\pythontest\omrphototest\D\omr\D86261\*",
                    r"d:\pythontest\omrphototest\D\omr\D86262\*",
                    r"d:\pythontest\omrphototest\D\omr\D86263\*",
                    r"d:\pythontest\omrphototest\D\omr\D86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+19*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 2)
    }
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 29,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 28,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form
def form_17cr_E():

    omr_location = [
                    r"d:\pythontest\omrphototest\E\omr\E86261\*",
                    r"d:\pythontest\omrphototest\E\omr\E86262\*",
                    r"d:\pythontest\omrphototest\E\omr\E86263\*",
                    r"d:\pythontest\omrphototest\E\omr\E86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+17*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 2)
    }
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 28,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 27,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form
def form_17cr_F():

    omr_location = [
                    r"d:\pythontest\omrphototest\F\omr\F86261\*",
                    r"d:\pythontest\omrphototest\F\omr\F86262\*",
                    r"d:\pythontest\omrphototest\F\omr\F86263\*",
                    r"d:\pythontest\omrphototest\F\omr\F86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+17*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 2)
    }
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 28,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 27,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_G():

    omr_location = [
                    r"d:\pythontest\omrphototest\G\omr\G86261\*",
                    r"d:\pythontest\omrphototest\G\omr\G86262\*",
                    r"d:\pythontest\omrphototest\G\omr\G86263\*",
                    r"d:\pythontest\omrphototest\G\omr\G86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+8*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 3)
    }
    group.update({
        j + 5 * h: [(j, 3+8 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 3)
        for h in range(3, 4)
    })
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_H():

    omr_location = [
                    r"d:\pythontest\omrphototest\H\omr\H86261\*",
                    r"d:\pythontest\omrphototest\H\omr\H86263\*",
                    r"d:\pythontest\omrphototest\H\omr\H86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 4+7*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 2)
    }
    group.update({
        j + 5 * h: [(j, 5+ 7 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(2, 4)
    })
    group.update({
        j + 20+ 5 * h: [(j + 5, 4 + 7 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 2)
    })
    group.update({
        j + 20 + 5 * h: [(j + 5, 5 + 7 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(2, 4)
    })

    group.update({
        j + 40+ 5 * h: [(j + 10, 4+ 7* h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 1)
    })
    group.update({
        j + 40 + 5 * h: [(j + 10, 4 + 7 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 2)
        for h in range(1, 2)
    })
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 30,
            'mark_row_number': 16,
            'mark_valid_area_col_start': 4,
            'mark_valid_area_col_end': 29,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 15},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_I():

    omr_location = [
                    r"d:\pythontest\omrphototest\I\omr\I86261\*",
                    r"d:\pythontest\omrphototest\I\omr\I86263\*",
                    r"d:\pythontest\omrphototest\I\omr\I86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+5*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 6)
    }
    group.update({
        j + 30 + 5 * h: [(j + 5, 3 + 6 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 5)
    })
    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 32,
            'mark_row_number': 11,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 31,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_J():
    omr_location = [
                    r"d:\pythontest\omrphototest\J\omr\J86261\*",
                    r"d:\pythontest\omrphototest\J\omr\J86262\*",
                    r"d:\pythontest\omrphototest\J\omr\J86263\*",
                    r"d:\pythontest\omrphototest\J\omr\J86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+6*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 5)
    }
    group.update({
        j + 25 + 5 * h: [(j + 5, 3 + 6 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 5)
    })
    group.update({
        j + 50: [(j + 10, 3), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
    })
    group.update({
        j + 50 + 5: [(j + 10, 3 + 6), 8, 'H', 'ABCDEFGH', 'S'] for j in range(1, 6)
    })

    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 16,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 15},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_K():

    omr_location = [
                    r"d:\pythontest\omrphototest\K\omr\K86261\*",
                    r"d:\pythontest\omrphototest\K\omr\K86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')

    group = {
        j+5*h:[(j, 3+5*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 6)
    }
    group.update({
        j + 30 + 5 * h: [(j + 5, 3 + 8 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })


    card_form = {
        'len':omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 32,
            'mark_row_number': 11,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 31,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10},
        'group_format': group,
        'image_clip':{
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form


def form_17cr_L():
    omr_location = [
        r"d:\pythontest\omrphototest\L\omr\L86261\*",
        r"d:\pythontest\omrphototest\L\omr\L86263\*",
        r"d:\pythontest\omrphototest\L\omr\L86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 11 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 3)
    }
    group.update({
        j + 15 + 5 * h: [(j + 5, 3 + 11 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 3)
    })
    group.update({
        j + 30 + 5 * h: [(j + 10, 3 + 11 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 3)
    })
    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 29,
            'mark_row_number': 16,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 28,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 15},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_M():
    omr_location = [
        r"d:\pythontest\omrphototest\M\omr\M86261\*",
        r"d:\pythontest\omrphototest\M\omr\M86262\*",
        r"d:\pythontest\omrphototest\M\omr\M86263\*",
        r"d:\pythontest\omrphototest\M\omr\M86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 6 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 5)
    }
    group.update({
        j + 25 + 5 * h: [(j + 5, 3 + 6 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 5)
    })
    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 11,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10,
            'mark_location_block_row': 11,
            'mark_location_block_col': 31
            },
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_N():
    omr_location = [
        r"d:\pythontest\omrphototest\N\omr\N86261\*",
        r"d:\pythontest\omrphototest\N\omr\N86262\*",
        r"d:\pythontest\omrphototest\N\omr\N86263\*",
        r"d:\pythontest\omrphototest\N\omr\N86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 8 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    }
    group.update({
        j + 20 + 5 * h: [(j + 5, 3 + 8 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 11,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_O():
    omr_location = [
        r"d:\pythontest\omrphototest\O\omr\O86261\*",
        r"d:\pythontest\omrphototest\O\omr\O86262\*",
        r"d:\pythontest\omrphototest\O\omr\O86263\*",
        r"d:\pythontest\omrphototest\O\omr\O86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 12 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 3)
    }

    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_P():
    omr_location = [
        r"d:\pythontest\omrphototest\P\omr\P86261\*",
        r"d:\pythontest\omrphototest\P\omr\P86262\*",
        r"d:\pythontest\omrphototest\P\omr\P86263\*",
        r"d:\pythontest\omrphototest\P\omr\P86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 8 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    }

    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form

def form_17cr_Q():
    omr_location = [
        r"d:\pythontest\omrphototest\Q\omr\Q86261\*",
        r"d:\pythontest\omrphototest\Q\omr\Q86262\*",
        r"d:\pythontest\omrphototest\Q\omr\Q86263\*",
        r"d:\pythontest\omrphototest\Q\omr\Q86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 5 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 2)
    }
    group.update({
        j + 5 * h: [(j, 3 + 5 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 3)
        for h in range(2, 3)
    })
    group.update({
        j + 5 * h: [(j, 4 + 5* h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(3, 5)
    })
    group.update({
        j + 5 * h: [(j, 4 + 5 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 3)
        for h in range(5, 6)
    })

    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 33,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 32,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form
def form_17cr_R():
    omr_location = [
        r"d:\pythontest\omrphototest\R\omr\R86261\*",
        r"d:\pythontest\omrphototest\R\omr\R86262\*",
        r"d:\pythontest\omrphototest\R\omr\R86263\*",
        r"d:\pythontest\omrphototest\R\omr\R86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 8 * h), 4, 'H', 'ABCD', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    }

    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 80,
            'y_end': -1
        }
    }
    return card_form


def form_17cr_S():
    omr_location = [
        r"d:\pythontest\omrphototest\S\omr\S86261\*",
        r"d:\pythontest\omrphototest\S\omr\S86262\*",
        r"d:\pythontest\omrphototest\S\omr\S86263\*",
        r"d:\pythontest\omrphototest\S\omr\S86268\*"]
    # omr_location = ['d:/work/omr_test_data']
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j+1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    }
    group.update({
        j + 20 + 5 * h: [(j + 6, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 40 + 5 * h: [(j + 11, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 60 + 5 * h: [(j + 15+1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 80 + 5 * h: [(j + 20 + 1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 5)
        for h in range(0, 1)
    })
    group.update({
        j + 84 + 5 * h: [(j + 24 + 1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 104 + 5 * h: [(j + 29 + 1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 5)
        for h in range(0, 1)
    })

    group.update({
        j + 108 + 5 * h: [(j + 33 + 1, 3 + 9 * h), 5, 'H', 'ABCDE', 'M'] for j in range(1, 6)
        for h in range(0, 2)
    })

    group.update({
        j + 118: [(j + 33 + 1, 21), 5, 'H', 'ABCDE', 'M'] for j in range(1, 3)
    })

    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 26,
            'mark_row_number': 40,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 25,
            'mark_valid_area_row_start': 2,
            'mark_valid_area_row_end': 39},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 10,
            'x_end': -25,
            'y_start': 330,
            'y_end': -1
        }
    }
    return card_form


def form_17cr_Sold():
    omr_location = [
        r"d:\pythontest\omrphototest\S\omr\S86261\*",
        r"d:\pythontest\omrphototest\S\omr\S86262\*",
        r"d:\pythontest\omrphototest\S\omr\S86263\*",
        r"d:\pythontest\omrphototest\S\omr\S86268\*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j, 3 + 6 * h), 5, 'H', 'ABCDE', 'M'] for j in range(1, 6)
        for h in range(0, 4)
    }
    group.update({
        j + 20 + 5 * h: [(j + 5, 3 + 6 * h), 5, 'H', 'ABCDE', 'M'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 40 + 5 * h: [(j + 10, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 60 + 5 * h: [(j + 15, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 80 + 5 * h: [(j + 20, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 5)
        for h in range(0, 1)
    })
    group.update({
        j + 84 + 5 * h: [(j + 24, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 104 + 5 * h: [(j + 29, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 5)
        for h in range(0, 1)
    })

    group.update({
        j + 108 + 5 * h: [(j + 33, 3 + 9 * h), 5, 'H', 'ABCDE', 'M'] for j in range(1, 6)
        for h in range(0, 2)
    })

    group.update({
        j + 118: [(j + 33, 21), 5, 'H', 'ABCDE', 'M'] for j in range(1, 3)
    })

    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 26,
            'mark_row_number': 40,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 25,
            'mark_valid_area_row_start': 2,
            'mark_valid_area_row_end': 38},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -1,
            'y_start': 330,
            'y_end': -1
        }
    }
    return card_form
