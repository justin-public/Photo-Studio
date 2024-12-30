from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import torch
import sys
import cv2
import numpy as np
import threading
import ftplib
import os
import qrcode
import time
import logging
from RealESRGAN import RealESRGAN
from PIL import Image
from PyQt5.QtMultimediaWidgets import QVideoWidget
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler,DPMSolverMultistepScheduler,StableDiffusionControlNetPipeline ,StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

HOST = "epicgramdev.cafe24.com"     # 192.168.0.114
ID = 'epicgramdev'
PW = 'dev@1357'

main_win = uic.loadUiType('PAGE/main_page.ui')[0]
layout_ui = uic.loadUiType('PAGE/LayOut.ui')[0]
layout_ui1 = uic.loadUiType('PAGE/LayOut1.ui')[0]
layout_ui2 = uic.loadUiType('PAGE/LayOut2.ui')[0]

count_stage_ = 0
time_val_ = 0
time_val_Capture = 0

class Layout2(QDialog,QWidget,layout_ui1):
    keycommand1 = QtCore.pyqtSignal(int)
    generatorcommand = QtCore.pyqtSignal(int)
    photoselectcommand1 = QtCore.pyqtSignal(int)
    photoselectcommand2 = QtCore.pyqtSignal(int)
    photoselectcommand3 = QtCore.pyqtSignal(int)
    homecommand = QtCore.pyqtSignal(int)
    
    def __init__(self):
        super(Layout2,self).__init__()
        self.initUI()
    def initUI(self):
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
    def UI_transition_sub(self,sel):
        if sel == 0:
            self.Photo1lb.close()
            self.Photo2lb.close()
            self.Photo3lb.close()

            self.Select1lb.close()
            self.Select2lb.close()
            self.Select3lb.close()
            self.homelb.close()    

            self.blacklb.setStyleSheet("background-color: black;")
            self.blacklb.show()
            
            pixmap = QtGui.QPixmap("resizesavephoto.png")
            self.Photolb.setPixmap(pixmap)
            self.Photolb.show()

            pixmap1 = QtGui.QPixmap("GUI/retake.png")
            self.retakelb.setPixmap(pixmap1)
            self.retakelb.mousePressEvent = self.Action_Select_sub
            self.retakelb.show()

            pixmap2 = QtGui.QPixmap("GUI/generator.png")
            self.generatorlb.setPixmap(pixmap2)
            self.generatorlb.mousePressEvent = self.Stable_diffusion_Process
            self.generatorlb.show()
        
        if sel == 1:
            self.Photolb.close()
            self.retakelb.close()
            self.generatorlb.close()
            
            self.movie = QMovie("GUI/loading.gif")
            self.giflb.setMovie(self.movie)
            self.giflb.show()
            self.movie.start()

            pixmap3 = QtGui.QPixmap("GUI/converting.png")
            self.Convertlb.setPixmap(pixmap3)
            self.Convertlb.show()

        if sel == 2:
            self.movie.stop()
            self.giflb.close()
            self.Convertlb.close()

            pixmapf = QtGui.QPixmap("real_size.png")
            self.Photo1lb.setPixmap(pixmapf)
            self.Photo1lb.show()

            pixmap5 = QtGui.QPixmap("real_size1.png")
            self.Photo2lb.setPixmap(pixmap5)
            self.Photo2lb.show()

            pixmap6 = QtGui.QPixmap("real_size2.png")
            self.Photo3lb.setPixmap(pixmap6)
            self.Photo3lb.show()

            pixmap7 = QtGui.QPixmap("GUI/select.png")
            self.Select1lb.setPixmap(pixmap7)
            self.Select1lb.mousePressEvent = self.Photo_Select_1
            self.Select1lb.show()

            pixmap8 = QtGui.QPixmap("GUI/select.png")
            self.Select2lb.setPixmap(pixmap8)
            self.Select2lb.mousePressEvent = self.Photo_Select_2
            self.Select2lb.show()

            pixmap9 = QtGui.QPixmap("GUI/select.png")
            self.Select3lb.setPixmap(pixmap9)
            self.Select3lb.mousePressEvent = self.Photo_Select_3
            self.Select3lb.show()
        
        if sel == 3:
            self.movie.stop()
            self.giflb.close()
            self.Convertlb.close()
            
            self.Photo1lb.close()
            self.Photo2lb.close()
            self.Photo3lb.close()
            self.Select1lb.close()
            self.Select2lb.close()
            self.Select3lb.close()

            pixmap10 = QtGui.QPixmap("GUI/qrBG.png")
            self.Qrbaselb.setPixmap(pixmap10)
            self.Qrbaselb.show()

            pixmap11 = QtGui.QPixmap("qrcode.png")
            self.Qrcodelb.setPixmap(pixmap11)
            self.Qrcodelb.show()

            pixmap12 = QtGui.QPixmap("GUI/home.png")
            self.homelb.setPixmap(pixmap12)
            self.homelb.mousePressEvent = self.homebutton
            self.homelb.show()
        
        if sel == 4:
            self.blacklb.close()
            self.Qrbaselb.close()
            self.Qrcodelb.close()
            self.homelb.close()
            
        if sel == 5:
            self.Photo1lb.close()
            self.Photo2lb.close()
            self.Photo3lb.close()
            self.Select1lb.close()
            self.Select2lb.close()
            self.Select3lb.close()

    QtCore.pyqtSlot(int)
    def Action_Select_sub(self,event):
        self.blacklb.close()
        self.Photolb.close()
        self.retakelb.close()
        self.generatorlb.close()
        self.keycmd = 1
        self.keycommand1.emit(self.keycmd)
    
    QtCore.pyqtSlot(int)
    def Stable_diffusion_Process(self,event):
        self.generatorcmd = 1
        self.generatorcommand.emit(self.generatorcmd)
    
    QtCore.pyqtSlot(int)
    def Photo_Select_1(self,event):
        self.photoselectcmd1 = 1
        self.photoselectcommand1.emit(self.photoselectcmd1)
    
    QtCore.pyqtSlot(int)
    def Photo_Select_2(self,event):
        self.photoselectcmd2 = 1
        self.photoselectcommand2.emit(self.photoselectcmd2)

    QtCore.pyqtSlot(int)
    def Photo_Select_3(self,event):
        self.photoselectcmd3 = 1
        self.photoselectcommand3.emit(self.photoselectcmd3)
    
    QtCore.pyqtSlot(int)
    def homebutton(self,event):
        self.homecmd = 1
        self.homecommand.emit(self.homecmd)


class Layout1(QDialog,QWidget,layout_ui):
    keycommand = QtCore.pyqtSignal(int)
    Photocommand = QtCore.pyqtSignal(bool)
    def __init__(self):
        super(Layout1,self).__init__()
        self.initUI()
    
    def initUI(self):
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
    
    def UI_transition(self,sel):
        if sel == 0:
            pixmap = QtGui.QPixmap("GUI/black_50.png")
            self.black_50_LY.setPixmap(pixmap)
            self.black_50_LY.show()
        
            pixmap1 = QtGui.QPixmap("GUI/guide.png")
            self.Guide_Ly.setPixmap(pixmap1)
            self.Guide_Ly.show()

            pixmap2 = QtGui.QPixmap("GUI/take.png")
            self.takelb.setPixmap(pixmap2)
            self.takelb.mousePressEvent = self.Action_Select
            self.takelb.show() 
            
        if sel == 1:
            self.black_50_LY.close()
            self.Guide_Ly.close()
            self.takelb.close()
            
            self.time_val_ = QtCore.QTimer(self)
            self.time_val_.setInterval(1000)
            self.time_val_.timeout.connect(self.time_count)
            self.time_val_.start()
        
    def time_count(self):
        global time_val_
        time_val_ += 1
        if time_val_ == 1:
            pixmap3 = QtGui.QPixmap("GUI/3.png")
            self.Countlb.setPixmap(pixmap3)
            self.Countlb.show()
        if time_val_ == 2:
            self.Countlb.close()
            pixmapf = QtGui.QPixmap("GUI/2.png")
            self.Countlb.setPixmap(pixmapf)
            self.Countlb.show()
        if time_val_ == 3:
            self.Countlb.close()
            pixmap5 = QtGui.QPixmap("GUI/1.png")
            self.Countlb.setPixmap(pixmap5)
            self.Countlb.show()
        if time_val_ == 4:
            self.Countlb.close()
            pixmap7 = QtGui.QPixmap("GUI/flash.png")
            self.black_50_LY.setPixmap(pixmap7)
            self.black_50_LY.show()
            self.Capture_active()        # 0.5 초 타이머 발동
            self.time_val_.stop()
            
        
    def Capture_active(self):
        self.time_val_Capture = QtCore.QTimer(self)
        self.time_val_Capture.setInterval(500)
        self.time_val_Capture.timeout.connect(self.Capture_time)
        self.time_val_Capture.start()
    
    QtCore.pyqtSlot(bool)
    def Capture_time(self):
        global time_val_Capture
        time_val_Capture += 1
        if time_val_Capture == 1:
            self.black_50_LY.close()
        
        if time_val_Capture == 2:
            self.Capture_cmd = True
            self.Photocommand.emit(self.Capture_cmd)
            self.time_val_Capture.stop()      # 0.5초 타이머 멈춤
            #self.time_val_Capture = 0
        
    QtCore.pyqtSlot(int)
    def Action_Select(self,event):
        self.keyint = 1
        self.keycommand.emit(self.keyint)


class VideoWork(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True
        self.Capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        
        self.cap_width = self.Capture.set(cv2.CAP_PROP_FRAME_WIDTH,910)
        self.cap_height = self.Capture.set(cv2.CAP_PROP_FRAME_HEIGHT,512)

        while self.ThreadActive:
            ret, frame = self.Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1920, 1080, Qt.IgnoreAspectRatio)
                self.ImageUpdate.emit(Pic)

        self.Capture.release()
        cv2.destroyAllWindows()
    
    def inverte2(self,imagem, name):
        for x in np.nditer(imagem, op_flags=['readwrite']):
            x=abs(x - 255)
        cv2.imwrite(name, imagem)

    def Image_capture(self):
        ret, img = self.Capture.read()
        if ret:
            cv2.flip(img,1)
            cv2.imwrite("savePhoto1.png", img)
            importimg = cv2.imread("savePhoto1.png")
            #gs_imagem = cv2.cvtColor(importimg,cv2.COLOR_BGR2GRAY)
            resizeimage = cv2.resize(importimg, (1250, 800))
            imageflip = cv2.flip(resizeimage, 1)
            #self.inverte2(resizeimage,"resizesavephoto.png") 
            cv2.imwrite("resizesavephoto.png",imageflip)

"""
class GifWork(QThread):
    dataLoaded = pyqtSignal(QByteArray)
    def __init__(self, all_widgets):
        QThread.__init__(self)
        self.all = all_widgets
        
    def run(self):
        time.sleep(1)
        f = QFile('GUI/loading.gif')
        f.open(f.ReadOnly)
        self.dataLoaded.emit(f.readAll())
        f.close()
"""

class MainWindow(QMainWindow,QDialog,QWidget,main_win):
    def __init__(self, sec = 0, parent = None):
        super(MainWindow,self).__init__(parent)
        self.state_check = 0
        self.initUI()
        self.keycmdpass()
        self.Photocmdpass()
        self.keycmdpass1()
        self.commandgeneratorpass()
        self.Photo1selectpass()
        self.Photo2selectpass()
        self.Photo3selectpass()
        self.homemovepass()

    def initUI(self):
        self.setupUi(self)
        self.camerathread = VideoWork()
        self.camerathread.start()
        self.camerathread.ImageUpdate.connect(self.ImageUpdateSlot)
        self.showFullScreen()
        self.layoutwin = Layout1()
        self.layoutwin1 = Layout2()
        
        self.layoutwin.UI_transition(self.state_check)
        self.layoutwin.showFullScreen()
        
    def keycmdpass(self):
        self.layoutwin.keycommand.connect(self.keycmd)
    
    def keycmd(self, sel):
        global count_stage_
        count_stage_ += sel
        if count_stage_ == 1:
            self.layoutwin.UI_transition(1)
        if count_stage_ == 2:
            self.layoutwin.UI_transition(2)
            
    def Photocmdpass(self):
        self.layoutwin.Photocommand.connect(self.Photocmd)  
    
    def Photocmd(self,sel):
        sel1 = sel
        if sel1 == True:
            self.camerathread.Image_capture()
            self.layoutwin1.UI_transition_sub(0)
            self.layoutwin1.showFullScreen()
    
    def keycmdpass1(self):
        self.layoutwin1.keycommand1.connect(self.keycmd1)
    
    def keycmd1(self,sel):
        global count_stage_
        global time_val_ 
        global time_val_Capture     
        sel1 = sel
        if sel1 == 1:
            count_stage_ = 0
            time_val_ = 0
            time_val_Capture = 0
            
            self.layoutwin.UI_transition(0)
            self.layoutwin.showFullScreen()
    
    def center_crop(img, dim):
        width, height = img.shape[1], img.shape[0]
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2)
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img
    
    #app = FaceAnalysis(name='buffalo_l')
    #app.prepare(ctx_id=0, det_size=(640, 640))
    #swapper = insightface.model_zoo.get_model('inswapper_128.onnx',download=True, download_zip=True)
    
    """
    def swap_n_snow(img1_fn,img2_fn, app, plot_before=True, plot_after=True):
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx',download=True, download_zip=True)
        
        img1 = cv2.imread(img1_fn)
        img2 = cv2.imread(img2_fn)
        
        if plot_before:
            fig, axs = plt.subplots(1,2,figsize=(10, 10))      # 10 , 5  
        face1 = app.get(img1)[0]
        face2 = app.get(img2)[0]

        img1_ = img1.copy()
        img2_ = img2.copy()
    
        if plot_after:
            img1_ = swapper.get(img1_, face1, face2, paste_back=True)
            img2_ = swapper.get(img2_, face2, face1, paste_back=True)
            
            cv2.imwrite("swap_image_test.png",img2_)
    """    
    
    # config file 을 정리해보자.
    def thread_stable_diffuser(self):
        input_image = cv2.imread("savePhoto1.png")
        if input_image is not None:
            with open('config02.txt','r') as file:
                config_data_line = file.read().splitlines()
            
            with open('config01.txt','r') as file1:
                info_data_line = file1.read().splitlines()
            
            HWNo = info_data_line[2]
            
            Model_ckpt = config_data_line[2]
            prompt = config_data_line[4]
            Negative_prompt = config_data_line[6]
            Sample_step = config_data_line[8]
            img_width = config_data_line[12]
            img_height = config_data_line[14]
            guidance_scale = config_data_line[17]
            Canny_low = config_data_line[19]
            Canny_max = config_data_line[21]
            insightface_logic = config_data_line[23]

            desired_width = 910
            desired_height = 512
            resized_image = cv2.resize(input_image, (desired_width, desired_height))
            cv2.imwrite("savePhoto2.png",resized_image)
            input_image1 = cv2.imread("savePhoto2.png")
            
            crop_img = input_image1[0:512, 190:700]   # 170     
            cv2.imwrite("CropPhoto.png",crop_img)
            
            timestamp = time.time()
            lt = time.localtime(timestamp)
            
            formatted = time.strftime("%y%m%d", lt)
            device_id = HWNo
            formatted_1 = time.strftime("%H%M%S", lt)
            
            import_Canny_folder = "canny/"
            importformatted = formatted+device_id+formatted_1+".png"
            import_Crop_folder = "crop/"
            
            Canny_img = load_image("CropPhoto.png")
            Canny_img.save(import_Crop_folder + importformatted)
            
            Canny_img = np.array(Canny_img)
            Canny_img = cv2.Canny(Canny_img,int(Canny_low),int(Canny_max))
            Canny_img = Canny_img[:, :,None]
            Canny_img = np.concatenate([Canny_img, Canny_img, Canny_img], axis=2)
            
            Canny_img_vi = Image.fromarray(Canny_img)
            Canny_img_vi.save(import_Canny_folder + importformatted)    
            
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
            pipe_controlnet = StableDiffusionControlNetPipeline.from_ckpt(
                Model_ckpt,
                controlnet=controlnet,
                safety_checker=False,
                torch_dtype=torch.float16,

            )
            pipe_controlnet.to("cuda")
            pipe_controlnet.scheduler = DPMSolverMultistepScheduler.from_config(pipe_controlnet.scheduler.config)
            #pipe_controlnet.enable_xformers_memory_efficient_attention()
            pipe_controlnet.enable_model_cpu_offload()
            generator_set = torch.manual_seed(2)
            
            result_img = pipe_controlnet(image=Canny_img_vi,
                    prompt = prompt,
                    negative_prompt = Negative_prompt,
                    generator=generator_set,
                    width = 512,
                    height = 512,
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(Sample_step)).images[0]
            
            result_img1 = pipe_controlnet(image=Canny_img_vi,
                    prompt = prompt,
                    negative_prompt = Negative_prompt,
                    generator=generator_set,
                    width = 512,
                    height = 512,
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(Sample_step)).images[0]

            result_img2 = pipe_controlnet(image=Canny_img_vi,
                    prompt = prompt,
                    negative_prompt = Negative_prompt,
                    generator=generator_set,
                    width = 512,
                    height = 512,
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(Sample_step)).images[0]
            
            result_img.save("real_size.png")
            result_img1.save("real_size1.png")
            result_img2.save("real_size2.png")
            
            if insightface_logic == "True":
                app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                app.prepare(ctx_id=0, det_size=(640, 640))     # 640, 640
                swapper = insightface.model_zoo.get_model('inswapper_128.onnx',download=True, download_zip=True)
            
                img1 = cv2.imread("CropPhoto.png")
                img2 = cv2.imread("real_size.png")
                img3 = cv2.imread("real_size1.png")
                img4 = cv2.imread("real_size2.png")

                if img2 is not None:
                    face1 = app.get(img1)[0]
                    face2 = app.get(img2)[0]
                    img1_ = img1.copy()
                    img2_ = img2.copy()
                    img1_ = swapper.get(img1_, face1, face2, paste_back=True)
                    img2_ = swapper.get(img2_, face2, face1, paste_back=True)

                    cv2.imwrite("swap_image_1.png",img2_)
                    src = cv2.imread('swap_image_1.png')
            
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = RealESRGAN(device, scale = 2)
                    model.load_weights('weights/RealESRGAN_x2.pth', download=True)

                    path_to_image = 'swap_image_1.png'
                    image = Image.open(path_to_image).convert('RGB')

                    sr_image = model.predict(image)
                    sr_image.save('model_size.png')
                    im = Image.open("model_size.png")
                    im1 = im.resize((512,512))
                    im1.save("real_size.png")
            
                if img3 is not None:
                    face1 = app.get(img1)[0]
                    face2 = app.get(img3)[0]
                    img1_ = img1.copy()
                    img2_ = img3.copy()
                    img1_ = swapper.get(img1_, face1, face2, paste_back=True)
                    img2_ = swapper.get(img2_, face2, face1, paste_back=True)
                    cv2.imwrite("swap_image_2.png",img2_)
                    src = cv2.imread('swap_image_2.png')
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = RealESRGAN(device, scale = 2)
                    model.load_weights('weights/RealESRGAN_x2.pth', download=True)
                    
                    path_to_image = 'swap_image_2.png'
                    image = Image.open(path_to_image).convert('RGB')

                    sr_image = model.predict(image)
                    sr_image.save('model_size1.png')
                    im = Image.open("model_size1.png")
                    im1 = im.resize((512,512))
                    im1.save("real_size1.png")
            
                if img4 is not None:
                    face1 = app.get(img1)[0]
                    face2 = app.get(img4)[0]
                    img1_ = img1.copy()
                    img2_ = img4.copy()
                    img1_ = swapper.get(img1_, face1, face2, paste_back=True)
                    img2_ = swapper.get(img2_, face2, face1, paste_back=True)
                    cv2.imwrite("swap_image_3.png",img2_)
                    src = cv2.imread('swap_image_3.png')
                
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = RealESRGAN(device, scale = 2)
                    model.load_weights('weights/RealESRGAN_x2.pth', download=True)
                    
                    path_to_image = 'swap_image_3.png'
                    image = Image.open(path_to_image).convert('RGB')

                    sr_image = model.predict(image)
                    sr_image.save('model_size2.png')
                    im = Image.open("model_size2.png")
                    im1 = im.resize((512,512))
                    im1.save("real_size2.png")

            input_image1 = cv2.imread("real_size2.png")
            if input_image1 is not None:
                self.layoutwin1.UI_transition_sub(2)
    
    
    def thread_stable_diffuser_rezize(self):
        with open('config02.txt','r') as file:
                config_data_line = file.read().splitlines()   # 12 , 14
        with open('config01.txt','r') as file1:
                info_data_line = file1.read().splitlines()
        HWNo = info_data_line[2]
        host = info_data_line[12]
        hostid = info_data_line[14]
        hostpw = info_data_line[16]
        hostfolder = info_data_line[18]
        hostaddress = info_data_line[10]

        scale_up = int(config_data_line[12])
        model_path = config_data_line[14]
        Command_image_save_path = config_data_line[25]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = RealESRGAN(device, scale = scale_up)
        model.load_weights(model_path, download=True)        

        path_to_image = 'real_size.png'
        image = Image.open(path_to_image).convert('RGB')
            
        sr_image = model.predict(image)
        sr_image.save('super_resolution.png')

        #sr_image_import = cv2.imread("model_size.png")

        timestamp = time.time()
        lt = time.localtime(timestamp)

        formatted = time.strftime("%y%m%d", lt)
        device_id = HWNo
        formatted_1 = time.strftime("%H%M%S", lt)

        import_photo_folder = "photo/"
        importformatted_1 = formatted+device_id+formatted_1+".png"

        import_photo = load_image("super_resolution.png")
        import_photo.save(import_photo_folder + importformatted_1)

        ftp = ftplib.FTP(host)
        ftp.login(hostid,hostpw)
        dir = ftp.dir()
        ftp.cwd(hostfolder)
        localpath = Command_image_save_path
        local_filename = os.path.join(localpath,importformatted_1)
        STORcmd= "STOR "+ importformatted_1
        with open(local_filename,'rb') as read_f:
            ftp.storbinary(STORcmd,read_f)
        ftp.quit()
            
        URLcmd= hostaddress + importformatted_1 
        img = qrcode.make(URLcmd)
        img.save("qrcode.png")
            
        import_QR_folder = "QR/"
        
        importformatted_2 = formatted+device_id+formatted_1+".png"
        import_qr = load_image("qrcode.png")
        import_qr.save(import_QR_folder + importformatted_2)
            
        self.layoutwin1.UI_transition_sub(3)
    
    def thread_stable_diffuser_rezize1(self):
        with open('config02.txt','r') as file:
                config_data_line = file.read().splitlines()   # 12 , 14
        with open('config01.txt','r') as file1:
                info_data_line = file1.read().splitlines()
        HWNo = info_data_line[2]
        host = info_data_line[12]
        hostid = info_data_line[14]
        hostpw = info_data_line[16]
        hostfolder = info_data_line[18]
        hostaddress = info_data_line[10]
        scale_up = int(config_data_line[12])
        model_path = config_data_line[14]
        Command_image_save_path = config_data_line[25]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = RealESRGAN(device, scale = scale_up)
        model.load_weights(model_path, download=True)        

        path_to_image = 'real_size1.png'
        image = Image.open(path_to_image).convert('RGB')
            
        sr_image = model.predict(image)
        sr_image.save('super_resolution.png')

        #sr_image_import = cv2.imread("model_size.png")

        timestamp = time.time()
        lt = time.localtime(timestamp)

        formatted = time.strftime("%y%m%d", lt)
        device_id = HWNo
        formatted_1 = time.strftime("%H%M%S", lt)

        import_photo_folder = "photo/"
        importformatted_1 = formatted+device_id+formatted_1+".png"

        import_photo = load_image("super_resolution.png")
        import_photo.save(import_photo_folder + importformatted_1)

        ftp = ftplib.FTP(host)
        ftp.login(hostid,hostpw)
        dir = ftp.dir()
        ftp.cwd(hostfolder)
        localpath = Command_image_save_path
        local_filename = os.path.join(localpath,importformatted_1)
        STORcmd= "STOR "+ importformatted_1
        with open(local_filename,'rb') as read_f:
            ftp.storbinary(STORcmd,read_f)
        ftp.quit()
            
        URLcmd= hostaddress + importformatted_1 
        img = qrcode.make(URLcmd)
        img.save("qrcode.png")
            
        import_QR_folder = "QR/"
        
        importformatted_2 = formatted+device_id+formatted_1+".png"
        import_qr = load_image("qrcode.png")
        import_qr.save(import_QR_folder + importformatted_2)
            
        self.layoutwin1.UI_transition_sub(3)
    
    def thread_stable_diffuser_rezize2(self):
        with open('config02.txt','r') as file:
                config_data_line = file.read().splitlines()   # 12 , 14
        with open('config01.txt','r') as file1:
                info_data_line = file1.read().splitlines()
        
        HWNo = info_data_line[2]
        host = info_data_line[12]
        hostid = info_data_line[14]
        hostpw = info_data_line[16]
        hostfolder = info_data_line[18]
        hostaddress = info_data_line[10]
        scale_up = int(config_data_line[12])
        model_path = config_data_line[14]
        Command_image_save_path = config_data_line[25]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = RealESRGAN(device, scale = scale_up)
        model.load_weights(model_path, download=True)        

        path_to_image = 'real_size2.png'
        image = Image.open(path_to_image).convert('RGB')
            
        sr_image = model.predict(image)
        sr_image.save('super_resolution.png')

        #sr_image_import = cv2.imread("model_size.png")

        timestamp = time.time()
        lt = time.localtime(timestamp)

        formatted = time.strftime("%y%m%d", lt)
        device_id = HWNo
        formatted_1 = time.strftime("%H%M%S", lt)

        import_photo_folder = "photo/"
        importformatted_1 = formatted+device_id+formatted_1+".png"

        import_photo = load_image("super_resolution.png")
        import_photo.save(import_photo_folder + importformatted_1)

        ftp = ftplib.FTP(host)
        ftp.login(hostid,hostpw)
        dir = ftp.dir()
        ftp.cwd(hostfolder)
        localpath = Command_image_save_path
        local_filename = os.path.join(localpath,importformatted_1)
        STORcmd= "STOR "+ importformatted_1
        with open(local_filename,'rb') as read_f:
            ftp.storbinary(STORcmd,read_f)
        ftp.quit()
            
        URLcmd= hostaddress + importformatted_1 
        img = qrcode.make(URLcmd)
        img.save("qrcode.png")
            
        import_QR_folder = "QR/"
        
        importformatted_2 = formatted+device_id+formatted_1+".png"
        import_qr = load_image("qrcode.png")
        import_qr.save(import_QR_folder + importformatted_2)
            
        self.layoutwin1.UI_transition_sub(3)

    def commandgeneratorpass(self):
        self.layoutwin1.generatorcommand.connect(self.commandgenerator)
    
    def commandgenerator(self,sel):
        sel1 = sel
        if sel1 == 1:
            self.layoutwin1.UI_transition_sub(1)
            x1 = threading.Thread(target=self.thread_stable_diffuser)
            x1.start()
    
    
    def Photo1selectpass(self):
        self.layoutwin1.photoselectcommand1.connect(self.Photo1select)
    
    def Photo1select(self,sel):
        sel1 = sel
        if sel1 == 1:
            with open('config02.txt','r') as file:
                Resize_commnad_line = file.read().splitlines()
            with open('config01.txt','r') as file1:
                info_data_line = file1.read().splitlines()
            HWNo = info_data_line[2]
            host = info_data_line[12]
            hostid = info_data_line[14]
            hostpw = info_data_line[16]
            hostfolder = info_data_line[18]
            hostaddress = info_data_line[10]
            
            Command_Resize_logic = Resize_commnad_line[10]
            Command_image_save_path = Resize_commnad_line[25]
            
            if Command_Resize_logic == "False":
                timestamp = time.time()
                lt = time.localtime(timestamp)
            
                formatted = time.strftime("%y%m%d", lt)
                device_id = HWNo
                formatted_1 = time.strftime("%H%M%S", lt)
            
                import_photo_folder = "photo/"
                importformatted_1 = formatted+device_id+formatted_1+".png"
            
                import_photo = load_image("real_size.png")
                import_photo.save(import_photo_folder + importformatted_1)
            
                ftp = ftplib.FTP(host)
                ftp.login(hostid,hostpw)
                dir = ftp.dir()
                ftp.cwd(hostfolder)
                localpath = Command_image_save_path
                local_filename = os.path.join(localpath,importformatted_1)
                STORcmd= "STOR "+ importformatted_1
                with open(local_filename,'rb') as read_f:
                    ftp.storbinary(STORcmd,read_f)
                ftp.quit()
            
                RRLcmd= hostaddress + importformatted_1 
                img = qrcode.make(RRLcmd)
                img.save("qrcode.png")
            
                import_QR_folder = "QR/"
            
                importformatted_2 = formatted+device_id+formatted_1+".png"
                import_qr = load_image("qrcode.png")
                import_qr.save(import_QR_folder + importformatted_2)
            
                self.layoutwin1.UI_transition_sub(3)
            
            if Command_Resize_logic == "True":
                self.layoutwin1.UI_transition_sub(5)
                self.layoutwin1.UI_transition_sub(1)

                x2 = threading.Thread(target=self.thread_stable_diffuser_rezize)
                x2.start()


    
    def Photo2selectpass(self):
        self.layoutwin1.photoselectcommand2.connect(self.Photo2select)
    
    def Photo2select(self,sel):
        sel1 = sel
        if sel1 == 1:
            with open('config02.txt','r') as file:
                Resize_commnad_line = file.read().splitlines()
            with open('config01.txt','r') as file1:
                info_data_line = file1.read().splitlines()
            HWNo = info_data_line[2]
            host = info_data_line[12]
            hostid = info_data_line[14]
            hostpw = info_data_line[16]
            hostfolder = info_data_line[18]
            hostaddress = info_data_line[10]
            Command_Resize_logic = Resize_commnad_line[10]
            Command_image_save_path = Resize_commnad_line[25]
            
            if Command_Resize_logic == "False":
                timestamp = time.time()
                lt = time.localtime(timestamp)
            
                formatted = time.strftime("%y%m%d", lt)
                device_id = HWNo
                formatted_1 = time.strftime("%H%M%S", lt)
            
                import_photo_folder = "photo/"
                importformatted_1 = formatted+device_id+formatted_1+".png"

                import_photo = load_image("real_size1.png")
                import_photo.save(import_photo_folder + importformatted_1)
            
                ftp = ftplib.FTP(host)
                ftp.login(hostid,hostpw)
                entries = ftp.nlst() # 디렉토리 목록을 보여줌
                print(entries)
                dir = ftp.dir()
                print(dir)
                ftp.cwd(hostfolder)
                localpath = Command_image_save_path
                local_filename = os.path.join(localpath,importformatted_1)
                STORcmd = "STOR "+ importformatted_1
            
                with open(local_filename,'rb') as read_f:
                    ftp.storbinary(STORcmd,read_f)
                ftp.quit()
            
                URLcmd = hostaddress + importformatted_1
                img = qrcode.make(URLcmd)
                img.save("qrcode.png")
            
                import_QR_folder = "QR/"
                importformatted_2 = formatted+device_id+formatted_1+".png"
                import_qr = load_image("qrcode.png")
                import_qr.save(import_QR_folder + importformatted_2)
                self.layoutwin1.UI_transition_sub(3)
        
            if Command_Resize_logic == "True":
                self.layoutwin1.UI_transition_sub(5)
                self.layoutwin1.UI_transition_sub(1)

                x2 = threading.Thread(target=self.thread_stable_diffuser_rezize1)
                x2.start()
    
    def Photo3selectpass(self):
        self.layoutwin1.photoselectcommand3.connect(self.Photo3select)
    
    def Photo3select(self,sel):
        sel1 = sel
        if sel1 == 1:
            with open('config02.txt','r') as file:
                Resize_commnad_line = file.read().splitlines()
            with open('config01.txt','r') as file1:
                info_data_line = file1.read().splitlines()
            HWNo = info_data_line[2]
            host = info_data_line[12]
            hostid = info_data_line[14]
            hostpw = info_data_line[16]
            hostfolder = info_data_line[18]
            hostaddress = info_data_line[10]
            Command_Resize_logic = Resize_commnad_line[10]
            Command_image_save_path = Resize_commnad_line[25]
            
            if Command_Resize_logic == "False":
                timestamp = time.time()
                lt = time.localtime(timestamp)
            
                formatted = time.strftime("%y%m%d", lt)
                device_id = HWNo
                formatted_1 = time.strftime("%H%M%S", lt)
            
                import_photo_folder = "photo/"
                importformatted_1 = formatted+device_id+formatted_1+".png"
            
                import_photo = load_image("real_size2.png")
                import_photo.save(import_photo_folder + importformatted_1)
            
                ftp = ftplib.FTP(host)
                ftp.login(hostid,hostpw)
                entries = ftp.nlst() # 디렉토리 목록을 보여줌
                dir = ftp.dir()
                ftp.cwd(hostfolder)
            
                localpath = Command_image_save_path
                local_filename = os.path.join(localpath,importformatted_1)
                STORcmd= "STOR "+ importformatted_1
            
                with open(local_filename,'rb') as read_f:
                    ftp.storbinary(STORcmd,read_f)
                ftp.quit()
                URLcmd= hostaddress + importformatted_1
                img = qrcode.make(URLcmd)
                img.save("qrcode.png")
            
                import_QR_folder = "QR/"
                importformatted_2 = formatted+device_id+formatted_1+".png"
                import_qr = load_image("qrcode.png")
                import_qr.save(import_QR_folder + importformatted_2)
            
                self.layoutwin1.UI_transition_sub(3)
            
            if Command_Resize_logic == "True":
                self.layoutwin1.UI_transition_sub(5)
                self.layoutwin1.UI_transition_sub(1)

                x2 = threading.Thread(target=self.thread_stable_diffuser_rezize2)
                x2.start()
    
    def homemovepass(self):
        self.layoutwin1.homecommand.connect(self.homemove)
    
    def homemove(self,sel):
        sel1 = sel
        global count_stage_
        global time_val_ 
        global time_val_Capture
        if sel1 == 1:
            count_stage_ = 0
            time_val_ = 0
            time_val_Capture = 0
            self.layoutwin1.UI_transition_sub(4)
            self.layoutwin.UI_transition(0)
            self.layoutwin.showFullScreen()
            
    def ImageUpdateSlot(self, Image):
        self.Videolb.setPixmap(QPixmap.fromImage(Image))

def main():
    with open('config01.txt','r') as file1:
        info_data_line = file1.read().splitlines()
    HWNo = info_data_line[2]
    host = info_data_line[12]
    hostid = info_data_line[14]
    hostpw = info_data_line[16]
    hostfolder = info_data_line[18]
    hostaddress = info_data_line[10]
    ftpfiledel = info_data_line[20]
    
    ftp = ftplib.FTP(host)
    ftp.login(hostid,hostpw)
    #ftp.mkd(hostfolder)
    
    if ftpfiledel == "True":
        #ftp = ftplib.FTP(host)
        #ftp.login(hostid,hostpw)
        entries = ftp.nlst() # 디렉토리 목록을 보여줌
        dir = ftp.dir()
        #ftp.mkd(hostfolder)
        ftp.cwd(hostfolder)
        
        """
        for something in ftp.nlst():
            try:
                ftp.delete(something)
            except:
                pass
        for entry in ftp.nlst():
            try:
                #ftp.mkd(hostfolder)
                ftp.rmd(entry)
            except:
                pass
        """
        #ftp.mkd(hostfolder)
    
    app = QApplication(sys.argv)
    screensize = app.desktop().availableGeometry().size()
    win = MainWindow(screensize)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()