import uuid
from werkzeug.utils import secure_filename 
import os

def save_img_to_buffer(file, buffer_dir):
    img_uuid = str(uuid.uuid4())

    filename = secure_filename(file.filename)
    extension = filename.split(".")[-1]

    if os.path.exists(buffer_dir) == False:
        os.mkdir(buffer_dir)

    filename = img_uuid + ".{}".format(extension)
    file.save(os.path.join(buffer_dir, filename))
    print("saved {}".format(filename))
    return img_uuid 
