import glob
from PIL import Image
user_preds = []
user_folders = glob.glob("Cluster_Dataset\\*[!.png]")
for user in user_folders:
    uname = user.split('\\')[1]
    print(uname,end=' => ')
    cropped_files = glob.glob(user+'\\cropped\\*')
    print('[',end='')
    res = 'no_helmet'
    for file in cropped_files:
        im = image.load_img(file)
        width,height = im.size
        f, e = os.path.splitext(file)
        im = im.crop((1, 1, width, height//4))
        imResize = im.resize((160,160), Image.ANTIALIAS)
        imResize = imResize.convert('RGB')
        #imResize.save(f+'_1.png', 'PNG', quality=100)
        img = image.img_to_array(imResize)
        im_f = np.expand_dims(img,axis=0)
        result = model.predict(im_f)
        if(int(result[0][0]) == 1):
            res = 'helmet'
        print(int(result[0][0]),end=' ')
    print(']',end=' => ')
    print(res,end='\n')