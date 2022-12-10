# Backend
### Pretrained model
put
[colornet_iter_76000.pth](https://drive.google.com/file/d/1OszomA-HnE1ss5hJ1lY40CqJsZJIrJoT/view?usp=sharing)
in `/checkpoints/video_moredata_l1`

put 
[vgg19_conv.pth](https://drive.google.com/file/d/1euCsIqTwc4EOYh-M-r_03gHo03MH6aIy/view?usp=sharing),
[vgg19_gray.pth](https://drive.google.com/file/d/1PO_PIW_hBQTWkxGzNnI0dQnEqdtjXw4D/view?usp=sharing)
in 
`/colorization/data`

### Requirements
`pip install -r requirements.txt`


### Run server

`gunicorn --bind 0.0.0.0:5000 wsgi:app`


### How to use
`Post` at `/colorization` with body including `ref_id(1~10) & image`

`Response` : Reference-based colorized image
