import argparse
import os
import torch
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from models.NonlocalNet import VGG19_pytorch, WarpNet
from models.ColorVidNet import ColorVidNet
from colorize_img import colorize_image

app = Flask(__name__)


@app.route('/colorization', methods=['POST'])
def img_save():
    # Get input image
    image_file = request.files.get('input', '')
    filename = secure_filename(image_file.filename)
    path, file_ext = os.path.splitext(filename)
    pathname = 'images/input/'
    os.makedirs(pathname, exist_ok=True)
    image_file.save(pathname + filename)
    filename_new = 'input' + file_ext
    os.rename(pathname + filename, pathname + filename_new)

    # Get ref image
    ref_option = request.form.get('ref_option', '')
    if ref_option == "0":  # Original ref image
        print("Colorize with original reference")
        ref_id = request.form.get('ref_id')
        ref_name = str(ref_id) + ".jpg"
    else:  # User ref image
        print("Colorize with user reference")
        user_ref_image = request.files.get('user_ref', '')
        filename = secure_filename(user_ref_image.filename)
        path, file_ext = os.path.splitext(filename)
        pathname = 'images/ref/'
        os.makedirs(pathname, exist_ok=True)
        user_ref_image.save(pathname + filename)
        filename_new = 'user_ref' + file_ext
        os.rename(pathname + filename, pathname + filename_new)
        ref_name = filename_new

    # Start colorization
    print("Colorizing ...")
    colorize_image(
        opt,
        opt.clip_path,
        os.path.join(opt.ref_path, ref_name),
        opt.output_path,
        nonlocal_net,
        colornet,
        vggnet,
    )
    print("Colorization done !")
    # Send response
    print("Sending response")
    return send_file('images/output/output.jpg', mimetype='image/jpeg')


if __name__ == "__main__":
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_propagate", default=False, type=bool)
    parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2])
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--clip_path", type=str, default="./images/input")
    parser.add_argument("--ref_path", type=str, default="./images/ref")
    parser.add_argument("--output_path", type=str, default="./images/output")
    opt = parser.parse_args()

    # Load networks & models
    print("Loading models ...")
    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))

    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")

    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path, map_location=torch.device('cpu')))
    colornet.load_state_dict(torch.load(color_test_path, map_location=torch.device('cpu')))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cpu()
    colornet.cpu()
    vggnet.cpu()

    print("Model loaded")

    # Start app
    print("Start server !")
    app.run()
