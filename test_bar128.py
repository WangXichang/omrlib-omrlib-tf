
import omrlib as ol

def test_barcode():
    import form_test as ftest
    f8 = ftest.form_8()
    bar = ol.Barcoder()
    bar.get_bar_image(f8.file_list[1], clip_top=100, clip_right=50, clip_left=550, clip_bottom=920)
    bar.get_bar_image01()
    bar.get_bar_width()
    print(bar.bar_wid_list)
    return bar


