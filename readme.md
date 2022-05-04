# Image table OCR in Excel file using PHP with Python

## Requirements for Windows
* PHP : >= 7.1 
* Python : 3.9.2
* Tesseract: [https://github.com/UB-Mannheim/tesseract/wiki#tesseract-installer-for-windows](https://github.com/UB-Mannheim/tesseract/wiki#tesseract-installer-for-windows)

#### Required Python Library

1. pip - 22.0.4
   
	`https://phoenixnap.com/kb/install-pip-windows`
   

2. cv2

	`pip install opencv-python`
   
	`if above package not installed than applay before this:
   		"pip install --user package" and "python -m pip install -U pip --user"`


3. numpy
	
	`pip install numpy`


4. pandas

	`pip install pandas`


5. matplotlib.pyplot

	`pip install matplotlib`

	`if above package not installed than applay before this: " python -m pip install --upgrade pip --user "`


6. pytesseract

	`pip install pytesseract`

	`if above package not installed than applay before this: " python -m pip install --upgrade pip --user "`

#### NOTE : If some reason it's not work than try to those below commands 
1. `pip install --user package`
2. `pip install Jinja2`
3. `pip install openpyxl`


---



## Example

1 : Import package

`from ocr import main`

2: Create Object and pass Source Image path & Destination output path

```

source_file = r'E:\laragon\www\demos\php-python-ocr-image-table\4.jpg'

destination_path = f"E:\laragon\www\demos\php-python-ocr-image-table\rendered_files\"

ocr_obj = main.Base(source_file, destination_path)

```

3: Getting All type JSON response

`ocr_obj.get_ocr_data()`

- NOTE : here get 5 different type of json response, if want specific type json response change this code function like below.

```
File: ocr/main.py

    def get_ocr_data(self):
        self.inverting_image()

        # Creating a dataframe of the generated OCR list
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center, finalboxes, outer = self.image_based_cell_box()

        arr = np.array(outer)
        dataframe = pd.DataFrame(arr.reshape(len(row), countcol))

        # NOTE : use any specific orient these one, ('columns', 'records', 'index', 'split', 'table')
        print(dataframe.to_json(orient='records'))             
```

4: Export to excel response

`ocr_obj.export_to_excel()`


5: Config Params
- If we want to Inverted image try to use this after creation object: `ocr_obj.inverted_img_write = True`
- If we want to Vertical image try to use this after creation object: `ocr_obj.vertical_img_write = True`
- If we want to Horizontal image try to use this after creation object: `ocr_obj.horizontal_img_write = True`
- If we want to Horizontal with Vertical image try to use this after creation object: `ocr_obj.horizontal_with_vertical_img_write = True`
