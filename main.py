import os
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/colorization', methods=['POST'])
def img_save():
    image_file = request.files.get('input', '')
    ref_id = request.form.get('ref_id')

    filename = secure_filename(image_file.filename)
    pathname = 'saved_img/'
    os.makedirs(pathname, exist_ok=True)
    image_file.save(pathname + filename)

    # return send_file(pathname+filename, mimetype='image/jpeg')
    return ref_id


if __name__ == "__main__":
    app.run()
