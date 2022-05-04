from ocr import main
source_file = r'E:\laragon\www\demos\php-python-ocr-image-table\4.jpg'
destination_path = f"E:\laragon\www\demos\php-python-ocr-image-table\\rendered_files\\"
ocr_obj = main.Base(source_file, destination_path)
ocr_obj.inverted_img_write = True

# For getting json response data
ocr_obj.get_ocr_data()
