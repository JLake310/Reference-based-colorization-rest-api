import os
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import numpy as np
import cv2

import colorization.main
from gaussian import get_gaussian
from colorization import *
app = Flask(__name__)


@app.route('/colorization', methods=['POST'])
def img_save():
    image_file = request.files.get('input', '')
    ref_id = request.form.get('ref_id')

    filename = secure_filename(image_file.filename)
    path, file_ext = os.path.splitext(filename)
    pathname = 'saved_img/'
    os.makedirs(pathname, exist_ok=True)
    image_file.save(pathname + filename)
    filename_new = 'input' + file_ext
    os.rename(pathname + filename, pathname + filename_new)

    return send_file('output.jpg', mimetype='image/jpeg')
    # return ref_id


if __name__ == "__main__":
    app.run()
