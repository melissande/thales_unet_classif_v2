#from tf_unet import unet, util, image_util
#import h5py
#import numpy as np
import unet
import load_imgs
import util
import image_util
import minidataset


##preparing training set if not done
#input_patches,label_patches=load_imgs.preprocess('../dataset_impl/', 4, 572,572,[450,450])
#load_imgs.prepare_data('../dataset_impl/patches4/',input_patches,label_patches,0.75)

#with h5py.File('../dataset_impl/patches4/train.h5', 'r') as hf:
#    data_train = np.array(hf.get('data'))
#    label_train = np.array(hf.get('label'))

##split in  and testset
data_train,label_train,data_test,label_test=minidataset.extract('../../dataset_impl/patches4',20,6,15)#0,0
data_provider=image_util.SimpleDataProvider(data_train, label_train, channels_in=5,channels_out=4, n_class =16 )



##setup & training
net = unet.Unet(channels_in=5,channels_out=4, n_class =16)
trainer = unet.Trainer(net, batch_size=1, optimizer="momentum")#10
path = trainer.train(data_provider, "prediction", training_iters=20, epochs=6) #51-100

#verification

#prediction = net.predict(path, data_test) #data=testset

#unet.error_rate(prediction, util.crop_to_shape(label_test, prediction.shape))

#modified through reshape
         
#true_y=util.to_rgb(util.crop_to_shape(label_test, prediction.shape)) 
#est_y=util.to_rgb(prediction)
#util.save_image(true_y, 'true_y_fin.jpg')
#util.save_image(est_y, 'est_y_fin.jpg')


#img = util.combine_img_prediction(data_test_x, label_test, prediction)
#util.save_image(img, "prediction.jpg")
