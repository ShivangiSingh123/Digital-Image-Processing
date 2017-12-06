import sys
from PyQt4 import uic,QtGui,QtCore
from PyQt4.QtGui import *
from matplotlib import image as i
from matplotlib import pyplot as plt
import cv2
import math
from PIL import Image
from scipy import stats
import colorsys
import numpy as np
from osgeo import gdal
from scipy import ndimage as nd
import skimage
from skimage import morphology as morph 
import scipy.signal
from skimage import transform 
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from sklearn.decomposition import PCA
from pylab import *
from skimage import data, io, color
import gdalnumeric
from skimage import data,exposure,img_as_float
import mpl_toolkits.mplot3d.axes3d as p3

#Name of the PyQt ui file
QtDesignerFile = "GUI.ui" 

Ui_MainWindow, QtBaseClass = uic.loadUiType(QtDesignerFile)
current=[]
selection = False
roi = []
#Mouse selection status
#Empty Region of Interest Python List 
#roi = [x1, y1, x2, y2]
  
class Window(QtGui.QMainWindow ,Ui_MainWindow):
    
    window=[]
    
    
    
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("SATTELITE IMAGE MANIPULATOR")
        self.setWindowIcon(QtGui.QIcon('logo1.png'))
        #Connecting different labels to their respective functions
        
        self.open.triggered.connect(self.Open)
        self.new_2.triggered.connect(self.New)
        self.saveAs.triggered.connect(self.SaveAs)
        self.quit_2.triggered.connect(self.close)
        self.canny.triggered.connect(self.Canny)
        self.laplace.triggered.connect(self.Laplace)
        self.sobel.triggered.connect(self.Sobel)
        self.prewitt.triggered.connect(self.Prewitt)
        self.robert.triggered.connect(self.Robert)
        self.hough_2.triggered.connect(self.Hough)
        self.layers.triggered.connect(self.Layers)
        self.merge.triggered.connect(self.Merge)
        self.statistics_2.mousePressEvent=self.Statistics
        self.meanfilter.triggered.connect(self.MeanFilter)
        self.medianfilter_2.triggered.connect(self.MedianFilter)
        self.modefilter.triggered.connect(self.GaussianFilter)
        self.linear.triggered.connect(self.LinearContrast)
        self.log.triggered.connect(self.LogContrast)
        self.inverse.triggered.connect(self.InverseContrast)
        self.power.triggered.connect(self.GammaContrast)
        self.hsv.triggered.connect(self.HSV)
        self.banding.triggered.connect(self.Banding)
        self.haze.triggered.connect(self.Haze)
        self.gabor.triggered.connect(self.Gabor)
        self.binary.triggered.connect(self.Threshbinary)
        self.binaryInv.triggered.connect(self.BinaryInversion)
        self.truncate.triggered.connect(self.Truncate)
        self.Tozero.triggered.connect(self.ToZero)
        self.zero.triggered.connect(self.Zero)
        self.region.triggered.connect(self.RegionGrowing)
        self.watershed.triggered.connect(self.Watershed)
        self.erosion.triggered.connect(self.Erosion)
        self.dilation.triggered.connect(self.Dilation)
        self.opening.triggered.connect(self.Opening)
        self.closing.triggered.connect(self.Closing)
        self.subset.triggered.connect(self.Subset)
        self.pca.triggered.connect(self.PCA)
        self.unsupervised.triggered.connect(self.Unsupervised)
        self.about_2.triggered.connect(self.About)
        
        
    #Function for browsing file

    def Open(self,event):
        self.image.clear()
        del current[:]
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File", "/","Image Files (*.png *.jpg *.tiff *.bmp *.tif)")
        if fileName:
            
            img=QtGui.QPixmap(fileName,"1")
            scaled_img=img.scaled(self.image.size(),QtCore.Qt.KeepAspectRatio)
            self.image.setPixmap(scaled_img)
            if img.isNull():
                QtGui.QMessageBox.information(self,"Message","Cannot load %s."%fileName)
                return
            img=str(fileName)
            current.append(img)
            
            
            

    #Function for opening new file
            
    def New(self,event):
        other=Window()
        Window.window.append(other)
        other.show()
        

    #Function for saving file
        
    def SaveAs(self,event):
        if current:
            image=current[0]
            filename=QtGui.QFileDialog.getSaveFileName(self)
            if filename: 
                self.image.pixmap().save(filename)
        else:
            QtGui.QMessageBox.information(self,"Message","Please open an image first")
            return


    #Function for Canny edge detection...A multi-stage algorithm

    def Canny(self,event):
        def message(self):
                text1,ok=QInputDialog.getText(self,"Minimum Value","Enter the Min value")
                if ok:
                    if text1:
                        text2,ok=QInputDialog.getText(self,"Maximum Value","Enter the Max value")
                        if text2:
                            n1=int(text1)
                            n2=int(text1)
                            edges = cv2.Canny(img,n1,n2)
                            #n1 and n2 are minimum and maximum value of thresholding 
                            plt.subplot(1,1,1),plt.imshow(edges,cmap = 'gray')
                            plt.title('Canny'), plt.xticks([]), plt.yticks([])
                            plt.show()
                            
                        else:
                            QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                            message(self)
                    else:
                        QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                        message(self)
        if current:   
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return








        
    #Function for Laplace edge detection
            
    def Laplace(self,event):
        if current:  
            im=current[0]
            img=i.imread(im,0)
            
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            cv2.imshow("Laplace",laplacian)
            cv2.waitKey(0)
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return


    #Function for Sobel edge detection
            
    def Sobel(self,event):   
        def message(self,img):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size ")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self,img)
                else:
                    n=int(text)
                    if n%2==0:
                        QtGui.QMessageBox.information(self,"Message","Please enter a valid window size (Odd number)")
                        message(self,img)
                    else:
                        # converting to gray scale
                        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # remove noise
                        img1=cv2.GaussianBlur(img,(n,n),0)
                        #  convolute with proper kernels
                        sobelx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=3)  # x
                        sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=3)  # y

                        cv2.imshow("Sobel X",sobelx)
                        cv2.imshow("Sobel Y",sobely)
                        cv2.waitKey(0)
                        #plt.subplot(1,2,1),plt.imshow(sobelx,cmap = 'gray')
                        #plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
                        #plt.subplot(1,2,2),plt.imshow(sobely,cmap = 'gray')
                        #plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
                        #plt.show()                
                                
        if current:
            im=current[0]
            img=i.imread(im,0)   
            message(self,img)
           
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return


    #Function for Prewitt edge detection

    def Prewitt(self,event):
        if current:
            image=current[0]
            im =Image.open(image).convert('L')
            width,height = im.size
            mat = im.load()
            prewittx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
            prewitty = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
            prewittIm = Image.new('L', (width,height))
            pixels = prewittIm.load()
            linScale = .3
                #For each pixel in the image
            for row in range(width-len(prewittx)):
                for col in range(height-len(prewittx)):
                    Gx = 0
                    Gy = 0
                    for k in range(len(prewittx)):
                        for j in range(len(prewitty)):
                            val = mat[row+k, col+j] * linScale
                            Gx += prewittx[k][j] * val
                            Gy += prewitty[k][j] * val
                    pixels[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))
            plt.subplot(1,1,1),plt.imshow(prewittIm,cmap = 'gray')
            plt.title('Prewitt'), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return


    #Function for Robert edge detection

    def Robert(self,event):
        if current:
            image=current[0]
            im =Image.open(image).convert('L')
            #im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            width,height = im.size
            mat = im.load()    
            robertsx = [[1,0],[0,-1]]
            robertsy = [[0,1],[-1,0]]
            robertIm = Image.new('L', (width,height))
            pixels = robertIm.load()
            linScale = .7
        #For each pixel in the image
            for row in range(width-len(robertsx)):
                for col in range(height-len(robertsy)):
                    Gx = 0
                    Gy = 0
                    for i in range(len(robertsx)):
                        for j in range(len(robertsy)):
                            val = mat[row+i, col+j] * linScale
                            Gx += robertsx[i][j] * val
                            Gy += robertsy[i][j] * val

                    pixels[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))
            plt.subplot(1,1,1),plt.imshow(robertIm,cmap = 'gray')
            plt.title('Robert'), plt.xticks([]), plt.yticks([])
            plt.show()

        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return


      #Function to implement hough transform
            
    def Hough(self,event):
        if current:
            im=current[0]
            img=i.imread(im,0)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,100,200,apertureSize = 3)

            #minLineLength - Minimum length of line. Line segments shorter than this are rejected.
            #maxLineGap - Maximum allowed gap between line segments to treat them as single line.
         
            minLineLength =input("Enter the minimum line length ")
            maxLineGap = input("Enter the maximum line gap ")
            threshold=input("Enter the threshold value ")
            #cv2.HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) 
            lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold,minLineLength,maxLineGap)
            for x in range(0, len(lines)):
                for x1,y1,x2,y2 in lines[x]:
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            plt.subplot(1,1,1),plt.imshow(img,cmap = 'gray')
            plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return


   #Function to open different layers of an image
            
    def Layers(self,event):
        if current:
            img=current[0]
            image=i.imread(img)
            channels=cv2.split(image)
            zero_channel=np.zeros_like(channels[0])
            red_img=cv2.merge([zero_channel,zero_channel,channels[2]])
            green_img=cv2.merge([zero_channel,channels[1],zero_channel])
            blue_img=cv2.merge([channels[0],zero_channel,zero_channel])
            cv2.imshow('Red channel',red_img)
            cv2.imshow('Green channel',green_img)
            cv2.imshow('Blue channel',blue_img)
            cv2.waitKey(0)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

   #Function to show different band combination of an image
            
    def Merge(self,event):
        if current:
            img=current[0]
            image=cv2.imread(img)
            r, g, b = cv2.split(image)
            items=('r,g,b','r,r,r','b,b,b','g,g,g','r,b,g','r,b,b','r,g,g','r,b,r','r,g,r','r,r,b','r,r,g','b,b,g','b,g,r','b,r,g','b,b,r','b,g,b','b,g,g','b,r,r','g,b,g','g,r,g','g,b,b','g,r,r','g,r,b','g,b,r','g,g,r','g,g,b','b,r,b')
            num1,ok=QInputDialog.getItem(self,"Layers to Colors","Select the color combination",items,0,True)
            if ok and num1:
                n1=num1
                if (n1 == 'r,g,b'):
                    im=cv2.merge((r,g,b))
                elif (n1 == 'r,r,r'):
                    im=cv2.merge((r, r, r))
                elif (n1 == 'b,b,b'):
                    im=cv2.merge((b, b, b))
                elif (n1 == 'g,g,g'):
                    im=cv2.merge((g, g, g))
                elif (n1 == 'r,b,g'):
                    im=cv2.merge((r, b, g))
                elif (n1 == 'r,b,b'):
                    im=cv2.merge((r, b, b))
                elif (n1 == 'r,g,g'):
                    im=cv2.merge((r, g, g))
                elif (n1 == 'r,b,r'):
                    im=cv2.merge((r, b, r))
                elif (n1 == 'r,g,r'):
                    im=cv2.merge((r, g, r))
                elif (n1 == 'r,r,b'):
                    im=cv2.merge((r, r, b))
                elif (n1 == 'r,r,g'):
                    im=cv2.merge((r, r, g))
                elif (n1 == 'b,b,g'):
                    im=cv2.merge((b, b, g))
                elif (n1 == 'b,g,r'):
                    im=cv2.merge((b, g, r))
                elif (n1 == 'b,r,g'):
                    im=cv2.merge((b, r, g))
                elif (n1 == 'b,b,r'):
                    im=cv2.merge((b, b, r))
                elif (n1 == 'b,g,b'):
                    im=cv2.merge((b, g, b))
                elif (n1 == 'b,g,g'):
                    im=cv2.merge((b, g, g))
                elif (n1 == 'b,r,r'):
                    im=cv2.merge((b, r, r))
                elif (n1 == 'g,b,g'):
                    im=cv2.merge((g, b, g))
                elif (n1 == 'g,r,g'):
                    im=cv2.merge((g, r, g))
                elif (n1 == 'g,b,b'):
                    im=cv2.merge((g, b, b))
                elif (n1 == 'g,r,r'):
                    im=cv2.merge((g, r, r))
                elif (n1 == 'g,r,b'):
                    im=cv2.merge((g, r, b))
                elif (n1 == 'g,b,r'):
                    im=cv2.merge((g, b, r))
                elif (n1 == 'g,g,r'):
                    im=cv2.merge((g, g, r))
                elif (n1 == 'g,g,b'):
                    im=cv2.merge((g, g, b))
                else:
                    im=cv2.merge((b, r, b))
                    
                cv2.imshow('Resulting Image',im)
                cv2.waitKey(0)        
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

   #Function for finding Statistics of an image

    def Statistics(self,event):
        if current:
            im=current[0]
            img=i.imread(im,0)
            
            #Function for finding mean of an image
            mean=img.mean()
            
            #Function for finding median of an image
            median=np.median(img)

            #Function for finding mode of an image
            mode=stats.mode(img)
        
            #Function for finding standard deviation of an image
            stddev = np.std(img)

            #Function for finding variance of an image
            var=np.var(img)
            QtGui.QMessageBox.about(self,"Statistics","Mean <br> %s.<br>Median <br> %s.<br>Mode <br> %s.<br>Standard Deviation <br> %s.<br>Variance <br> %s.<br>"%(mean,median,mode[0][0][0][0],stddev,var))
       

       

           #Function for showing histogram of an image
            quit_msg = "Show histogram of image?"
            reply = QtGui.QMessageBox.question(self, 'Message', 
                     quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

            if reply == QtGui.QMessageBox. No:
                return
                 
            else:
                color = ('b','g','r')  # a tuple
                for j,col in enumerate(color):
                    histr = cv2.calcHist([img],[j],None,[256],[0,256])
                    plt.plot(histr,color = col)
                    plt.xlim([0,256])   #setting limit of x axis in matplotlib
                plt.show()
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    #Function for mean filtering of an image

    def MeanFilter(self,event):
        def message(self):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self)
                else:
                    n=int(text)
                    blur=cv2.blur(img,(n,n))
                    plt.subplot(1,1,1),plt.imshow(blur,cmap = 'gray')
                    plt.title('Mean Filtering'), plt.xticks([]), plt.yticks([])
                    plt.show()
        if current:
            im=current[0]
            img=i.imread(im,0)
            message(self)
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return


    #Function for median filtering of an image

    def MedianFilter(self,event):
        def message(self):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self)
                else:
                    n=int(text)
                    if n%2==0:
                       QtGui.QMessageBox.information(self,"Message","Please enter a valid window size (Odd number)")
                       message(self)
                    else:    
                        blur=cv2.medianBlur(img,n)
                        plt.subplot(1,1,1),plt.imshow(blur,cmap = 'gray')
                        plt.title('Median Filtering'), plt.xticks([]), plt.yticks([])
                        plt.show()
        if current:
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return
            
    #Function for  filtering of an image

    def GaussianFilter(self,event):
        def message(self):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self)
                else:
                    n=int(text)
                    if n%2==0:
                        QtGui.QMessageBox.information(self,"Message","Please enter a valid window size (Odd number)")
                        message(self)
                    else:
                        blur_img=cv2.GaussianBlur(img,(n,n),0)
                        plt.subplot(1,1,1),plt.imshow(blur_img,cmap = 'gray')
                        plt.title('Gaussian Filtering'), plt.xticks([]), plt.yticks([])
                        plt.show()

        if current:
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def LinearContrast(self,event):
        if current:
            im=current[0]
            img=i.imread(im,0)
            width,height=img.shape[:2]
            NewImg=np.zeros_like(img)
            InputMax=np.amax(img)
            InputMin=np.amin(img)
            a=(255.0/(InputMax-InputMin))
            b=255.0-(a*InputMax)
            for j in range(width):
                for k in range(height):
                
                    NewImg[j,k]=(a*(img[j,k]+b))
            plt.subplot(1,1,1),plt.imshow(NewImg,cmap = 'gray')
            plt.title('Linear Contrast Enhancement'), plt.xticks([]), plt.yticks([])
            plt.show()
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def LogContrast(self,event):
        if current:
            im=current[0]
            img=cv2.imread(im)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=img_as_float(img)
            logarithmic=exposure.adjust_log(img,1)
            #skimage.exposure.adjust_log(image, gain=1, inv=False)
            plt.subplot(1,1,1),plt.imshow(logarithmic,cmap = 'gray')
            plt.title('Logarithmic Contrast Enhancement'), plt.xticks([]), plt.yticks([])
            plt.show()
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def InverseContrast(self,event):
        if current:
            im=current[0]
            img=cv2.imread(im)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=img_as_float(img)
            logarithmic=exposure.adjust_log(img,1,inv=True)
        
            plt.subplot(1,1,1),plt.imshow(logarithmic,cmap = 'gray')
            plt.title('Inverse Logarithmic Contrast Enhancement'), plt.xticks([]), plt.yticks([])
            plt.show()
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def GammaContrast(self,event):
        if current:
            im=current[0]
            img=cv2.imread(im)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=img_as_float(img)
            gamma=exposure.adjust_gamma(img,1)
            #skimage.exposure.adjust_gamma(image, gamma=1, gain=1)
            plt.subplot(1,1,1),plt.imshow(gamma,cmap = 'gray')
            plt.title('Power Contrast Enhancement'), plt.xticks([]), plt.yticks([])
            plt.show()
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return 

    def Gabor(self):
        # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        # ksize - size of gabor filter (n, n)
        # sigma - standard deviation of the gaussian function
        # theta - orientation of the normal to the parallel stripes
        # lambda - wavelength of the sunusoidal factor
        # gamma - spatial aspect ratio
        # psi - phase offset
        # ktype - type and range of values that each pixel in the gabor kernel can hold
          
        def build_filters():
            filters=[]
            sigma=input("Enter the value for sigma ")
            lam=input("Enter the value for lambda ")
            gamma=input("Enter the value for gamma ")
            psi=input("Enter the value for phase offset ")
            for theta in np.arange(0,np.pi,np.pi/16):
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, psi, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
            return filters

        def process(img,filters):
            accum=np.zeros_like(img)
            for kern in filters:
                fimg=cv2.filter2D(img,cv2.CV_8UC3,kern)
                np.maximum(accum,fimg,accum)
            return accum
        if current:
            im=current[0]
            img=cv2.imread(im)
            ksize=input("Enter the kernel size (odd value)")
            filters = build_filters()
            res1 = process(img, filters)
            cv2.imshow('result', res1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    

    def HSV(self,event):
        if current:
            im=current[0]
            img=cv2.imread(im)
            HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            channels=cv2.split(HSV)
            hue=cv2.merge([channels[0]])
            saturation=cv2.merge([channels[1]])
            intensity=cv2.merge([channels[2]])
            cv2.imshow("HSV",HSV)
            cv2.imshow('Hue',hue)
            cv2.imshow('Saturation',saturation)
            cv2.imshow('Intensity',intensity)
            cv2.waitKey(0)
            quit_msg = "Show histogram of HSV image?"
            reply = QtGui.QMessageBox.question(self, 'Message', 
                     quit_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

            if reply == QtGui.QMessageBox. No:
                return
                 
            else:
                xs,ys=img.shape[:2]
                max_intensity=100
                hues={}

                for x in range(0,xs):
                    for y in range(0,ys):
                        [r,g,b]=img[x,y]
                        r/=255.0
                        g/=255.0
                        b/=255.0
                        [h,s,v]=colorsys.rgb_to_hsv(r,g,b)

                        if h not in hues:
                            hues[h]={}

                        if v not in hues[h]:
                            hues[h][v]=1
                        else:
                            if hues[h][v]<max_intensity:
                                hues[h][v]+=1

                h_=[]
                v_=[]
                i=[]
                colors=[]

                for h in hues:  
                    for v in hues[h]:
                        h_.append(h)
                        v_.append(v)
                        i.append(hues[h][v])
                        [r,g,b]=colorsys.hsv_to_rgb(h,1,v)
                        colors.append([r,g,b])
       
                fig=plt.figure()
                ax=fig.add_subplot(111,projection='3d')
                ax.scatter(h_,v_,i,s=5,c=colors,lw=0)
                ax.set_xlabel('Hue')
                ax.set_ylabel('Value')
                ax.set_zlabel('Intensity')
                fig.add_axes(ax)
                plt.show()
            

            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return
        
    #Function for correction of line banding
        
    def Banding(self,event):
        def destripe(image, height, width, horizontal=True):
   
                if horizontal:
                    lowbox = (skimage.morphology.rectangle(height,width)).astype(np.float32)
                    lowbox[:] = 0.5 / (lowbox.shape[0]*lowbox.shape[1] - 1)
                    lowbox[lowbox.shape[0]/2,lowbox.shape[1]/2] = 0.5
                    lowpass = scipy.signal.convolve2d(image,lowbox,mode='same')
        
                    hibox = (skimage.morphology.rectangle(height/2+1,width/2+1)).astype(np.float32)
                    hibox[:] = 0.5 / (hibox.shape[0]*hibox.shape[1] - 1)
                    hibox[hibox.shape[0]/2,hibox.shape[1]/2] = 0.5
                    hipass = image - scipy.signal.convolve2d(image,hibox,mode='same')
                else:
                    lowbox = (skimage.morphology.rectangle(width,height)).astype(np.float32)
                    lowbox[:] = 0.5 / (lowbox.shape[0]*lowbox.shape[1] - 1)
                    lowbox[lowbox.shape[0]/2,lowbox.shape[1]/2] = 0.5
                    lowpass = scipy.signal.convolve2d(image,lowbox,mode='same')
        
                    hibox = (skimage.morphology.rectangle(width/2+1,height/2+1)).astype(np.float32)
                    hibox[:] = 0.5 / (hibox.shape[0]*hibox.shape[1] - 1)
                    hibox[hibox.shape[0]/2,hibox.shape[1]/2] = 0.5
                    hipass = image - scipy.signal.convolve2d(image,hibox,mode='same')
    
                dst = image - lowpass + hipass
                return(dst)
        if current:
            im=current[0]
            lo = gdal.Open(im)
            loband = lo.GetRasterBand(1)
            loarr = loband.ReadAsArray()
            loarr = skimage.img_as_float(loarr)
            

            dst_image = destripe(loarr,1,291,True)
            plt.subplot(1,1,1),plt.imshow(dst_image,cmap = 'gray')
            plt.title('Corrected Image'), plt.xticks([]), plt.yticks([])

            plt.show()

        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return 
        
    def Haze(self,event):
        if current:
            im=current[0]
            img=cv2.imread(im)
            width,height=img.shape[:2]
            NewImg=np.zeros_like(img)
            InputMin=np.amin(img)
            for j in range(width):
                for k in range(height):
                    x=img[j,k]
                    NewImg[j,k]=x-InputMin

            plt.subplot(1,1,1),plt.imshow(NewImg,cmap = 'gray')
            plt.title('Haze Correction'), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return





    def Threshbinary(self,event):
        def message(self):
                text1,ok=QInputDialog.getText(self,"Minimum Value","Enter the Min value")
                if ok:
                    if text1:
                        text2,ok=QInputDialog.getText(self,"Maximum Value","Enter the Max value")
                        if text2:
                            n1=int(text1)
                            n2=int(text1)
                            img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
                            ret,thresh=cv2.threshold(img1,n1,n2,cv2.THRESH_BINARY)
                            plt.subplot(1,1,1),plt.imshow(thresh,cmap = 'gray')
                            plt.title('Binary Threshold'), plt.xticks([]), plt.yticks([])
                            plt.show()
                            
                        else:
                            QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                            message(self)
                    else:
                        QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                        message(self)
        if current:   
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return









    def BinaryInversion(self,event):
        def message(self):
            text1,ok=QInputDialog.getText(self,"Minimum Value","Enter the Min value")
            if ok:
                if text1:
                    text2,ok=QInputDialog.getText(self,"Maximum Value","Enter the Max value")
                    if text2:
                        n1=int(text1)
                        n2=int(text1)
                        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
                        ret,thresh=cv2.threshold(img1,n1,n2,cv2.THRESH_BINARY_INV)
                        plt.subplot(1,1,1),plt.imshow(thresh,cmap = 'gray')
                        plt.title('Binary Inversion Threshold'), plt.xticks([]), plt.yticks([])
                        plt.show()
                            
                    else:
                        QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                        message(self)
                else:
                    QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                    message(self)
        if current:   
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return










        

    def Truncate(self,event):
        def message(self):
            text1,ok=QInputDialog.getText(self,"Minimum Value","Enter the Min value")
            if ok:
                if text1:
                    text2,ok=QInputDialog.getText(self,"Maximum Value","Enter the Max value")
                    if text2:
                        n1=int(text1)
                        n2=int(text1)
                        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
                        ret,thresh=cv2.threshold(img1,n1,n2,cv2.THRESH_TRUNC)
                        plt.subplot(1,1,1),plt.imshow(thresh,cmap = 'gray')
                        plt.title('Truncate'), plt.xticks([]), plt.yticks([])
                        plt.show()
                            
                    else:
                        QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                        message(self)
                else:
                    QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                    message(self)
        if current:   
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return








        

    def ToZero(self,event):
        def message(self):
            text1,ok=QInputDialog.getText(self,"Minimum Value","Enter the Min value")
            if ok:
                if text1:
                    text2,ok=QInputDialog.getText(self,"Maximum Value","Enter the Max value")
                    if text2:
                        n1=int(text1)
                        n2=int(text1)
                        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
                        ret,thresh=cv2.threshold(img1,n1,n2,cv2.THRESH_TOZERO)
                        plt.subplot(1,1,1),plt.imshow(thresh,cmap = 'gray')
                        plt.title('To Zero Image'), plt.xticks([]), plt.yticks([])
                        plt.show()
                            
                    else:
                        QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                        message(self)
                else:
                    QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                    message(self)
        if current:   
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return








        

    def Zero(self,event):
        def message(self):
            text1,ok=QInputDialog.getText(self,"Minimum Value","Enter the Min value")
            if ok:
                if text1:
                    text2,ok=QInputDialog.getText(self,"Maximum Value","Enter the Max value")
                    if text2:
                        n1=int(text1)
                        n2=int(text1)
                        img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
                        ret,thresh=cv2.threshold(img1,n1,n2,cv2.THRESH_TOZERO_INV)
                        plt.subplot(1,1,1),plt.imshow(thresh,cmap = 'gray')
                        plt.title('To Zero inverted Image'), plt.xticks([]), plt.yticks([])
                        plt.show()
                            
                    else:
                        QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                        message(self)
                else:
                    QtGui.QMessageBox.information(self,"Message","Please enter both the minimum and maximum values")
                    message(self)
        if current:   
            im=current[0]
            img=i.imread(im,0)
            message(self)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return






        
    def RegionGrowing(self, event):
        def get8n(x, y, shape):
            out = []
            maxx = shape[1]-1
            maxy = shape[0]-1

           #top left
            outx = min(max(x-1,0),maxx)
            outy = min(max(y-1,0),maxy)
            out.append((outx,outy))

           #top center
            outx = x
            outy = min(max(y-1,0),maxy)
            out.append((outx,outy))

            #top right
            outx = min(max(x+1,0),maxx)
            outy = min(max(y-1,0),maxy)
            out.append((outx,outy))

            #left
            outx = min(max(x-1,0),maxx)
            outy = y
            out.append((outx,outy))

            #right
            outx = min(max(x+1,0),maxx)
            outy = y
            out.append((outx,outy))

            #bottom left
            outx = min(max(x-1,0),maxx)
            outy = min(max(y+1,0),maxy)
            out.append((outx,outy))

            #bottom center
            outx = x
            outy = min(max(y+1,0),maxy)
            out.append((outx,outy))
            #bottom right
            outx = min(max(x+1,0),maxx)
            outy = min(max(y+1,0),maxy)
            out.append((outx,outy))
            return out
        def on_mouse(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print 'Seed: ' + str(x) + ', ' + str(y), img[y,x]
                clicks.append((y,x))
        
        def Region(img,seed):        
            list = []
            outimg = np.zeros_like(img)
            list.append((seed[0], seed[1]))
            processed = []
            while(len(list) > 0):
                pix = list[0]
                outimg[pix[0], pix[1]] = 255
                for coord in get8n(pix[0], pix[1], img.shape):
                    if img[coord[0], coord[1]] != 0:
                        outimg[coord[0], coord[1]] = 255
                        if not coord in processed:
                            list.append(coord)
                        processed.append(coord)
                list.pop(0)
       # cv2.imshow("Progress",outimg)
       # cv2.waitKey(1)
            return outimg
        clicks=[]
        if current:   
            im=current[0]
            image=cv2.imread(im,0)
            ret, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            cv2.namedWindow('Input')
            cv2.setMouseCallback('Input', on_mouse, 0, )
            cv2.imshow('Input', img)
            cv2.waitKey(0)
            seed=clicks[-1]
            out=Region(img,seed)
            cv2.imshow('Region Growing', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return



        
 
    def Watershed(self,event):
        if current:   
            im=current[0]
            img=i.imread(im,0)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)
            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0
            markers = cv2.watershed(img,markers)
            img[markers == -1] = [255,0,0]
            img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.namedWindow("Watershed Image")
            cv2.imshow('Watershed Image',img)
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

            
    def Erosion(self,event):
        def message(self):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self)
                else:
                    n=int(text)
                    kernel = np.ones((n,n), np.uint8)
                    img_erosion = cv2.erode(img, kernel, iterations=1)
                    cv2.imshow('Erosion', img_erosion)
                    cv2.waitKey(0)
                    
            
        if current:
            im=current[0]
            img=i.imread(im,0)
            message(self)
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return
        
    def Dilation(self,event):
        def message(self):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self)
                else:
                    n=int(text)
                    kernel = np.ones((n,n), np.uint8)
                    img_dilation = cv2.dilate(img, kernel, iterations=1)
                    cv2.imshow('Dilation', img_dilation)
                    cv2.waitKey(0)
                    
        if current:
            im=current[0]
            img=i.imread(im,0)
            message(self)
            
        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def Opening(self,event):
         def message(self):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self)
                else:
                    n=int(text)
                    kernel = np.ones((n,n), np.uint8)
                    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
                    cv2.imshow('Opening', opening)
                    cv2.waitKey(0)
         if current:
            im=current[0]
            img=i.imread(im,0)
            message(self)
            
         else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def Closing(self,event):
          def message(self):
            text,ok=QInputDialog.getText(self,"Window size","Enter the Window size")
            if ok:
                if text.isNull():
                    QtGui.QMessageBox.information(self,"Message","Please enter a value")
                    message(self)
                else:
                    n=int(text)
                    kernel = np.ones((n,n), np.uint8)
                    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                    cv2.imshow('Closing', closing)
                    cv2.waitKey(0)
                    
          if current:
            im=current[0]
            img=i.imread(im,0)
            message(self)
          else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return
        
    def Subset(self,event):
        def roi_selection(event, x, y, flags, param):
            #Reference to the global variables
            global roi,selection
            #On Left mouse button click records roi with mouse selection status to True
            if event == cv2.EVENT_LBUTTONDOWN:
                selection = True
                roi = [x, y, x, y]
            #On Mouse movement records roi with mouse selection status to True
            elif event == cv2.EVENT_MOUSEMOVE:
                if selection == True:
                    roi[2] = x
                    roi[3] = y          
            #If Left mouse button is released changes mouse selection status to False
            elif event == cv2.EVENT_LBUTTONUP:
                selection = False
                roi[2] = x
                roi[3] = y 
              
        if current:
            
            img=current[0]
            
            window_name='Input Image'

        #Cropped Image Window Name
            window_crop_name='Cropped Image'

        #Escape ASCII Keycode
            esc_keycode=27

        #Time to waitfor
            wait_time=1

       #Load an image
#cv2.IMREAD_COLOR = Default flag for imread. Loads color image.
#cv2.IMREAD_GRAYSCALE = Loads image as grayscale.
#cv2.IMREAD_UNCHANGED = Loads image which have alpha channels.
#cv2.IMREAD_ANYCOLOR = Loads image in any possible format
#cv2.IMREAD_ANYDEPTH = Loads image in 16-bit/32-bit otherwise converts it to 8-bit
            input_img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
            print "Drag the cursor and select the region you want to subset and press escape key and the resulting image will be saved in your current directory." 

#Check if image is loaded 
            if input_img is not None:
    # Make a copy of original image for cropping
                clone = input_img.copy()
    #Create a Window
    #cv2.WINDOW_NORMAL = Enables window to resize.
    #cv2.WINDOW_AUTOSIZE = Default flag. Auto resizes window size to fit an image.
                cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)
                global roi,selection
    #Set mouse handler for Window with roi_selection function callback
                cv2.setMouseCallback(window_name, roi_selection)

    #Loop 
                while True:
        #Show original image in window
                    cv2.imshow(window_name,input_img)
        
        #if roi has all parameters filled
                    if len(roi)==4:
            #Make a copy of orginal image before drawing rectangle on it
                        input_img = clone.copy()
            #Check if any pixl coorinalte is negative and make it zero
                        roi = [0 if j < 0 else j for j in roi]
            #Draw rectangle on input_img
            #input_image: source image
            #(roi[0], roi[1]): Vertex of the rectangle
            #(roi[2], roi[3]): Opposite Vertex of the rectangle
            #(0, 255, 0): Rectangular Color
            # 2: Thickness
                        cv2.rectangle(input_img, (roi[0],roi[1]), (roi[2],roi[3]), (0, 255, 0), 2)  
            #Make x and y coordiates for cropping in ascending order
            #if x1 = 200,x2= 10 make x1=10,x2=200
                        if roi[0] > roi[2]:
                            x1 = roi[2]
                            x2 = roi[0]
            #else keep it as it is  
                        else:
                            x1 = roi[0]
                            x2 = roi[2]
            #if y1 = 200,y2= 10 make y1=10,y2=200   
                        if roi[1] > roi[3]:
                            y1 = roi[3]
                            y2 = roi[1]
            #else keep it as it is  
                        else:
                            y1 = roi[1]
                            y2 = roi[3] 
                
            #Crop clone image
                        crop_img = clone[y1 : y2 , x1 : x2]
            #check if crop_img is not empty
                        if len(crop_img):
                #Create a cropped image Window
                            cv2.namedWindow(window_crop_name,cv2.WINDOW_AUTOSIZE)
                #Show image in window
                            cv2.imshow(window_crop_name,crop_img)
                            cv2.imwrite("crop_img.jpeg",crop_img)
           #Check if any key is pressed
                    k = cv2.waitKey(wait_time)
                    
           #Check if ESC key is pressed. ASCII Keycode of ESC=27
                    if k == esc_keycode:
                        
                        
                            
            #Destroy All Windows
                        cv2.destroyAllWindows()
                        break
                    
              
            else:
                print 'Please Check The Path of Input File'
        

        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def PCA(self,event):
        if current:
            img=current[0]
            img_gray = io.imread(img,as_grey=True)
            subplot(2, 2, 1)
            io.imshow(img_gray)
            xlabel('Original Image')
            n=input("Enter the range ")
            
            for i in range(1, n+1):
                n_comp = 5 ** i
                pca = PCA(n_components = n_comp)
                pca.fit(img_gray)
                img_gray_pca = pca.fit_transform(img_gray)
                img_gray_restored = pca.inverse_transform(img_gray_pca)
                subplot(2, 2, i+1)
                io.imshow(img_gray_restored)
                xlabel('Restored image n_components = %s' %n_comp)
                print 'Variance retained %s %%' %((1 - sum(pca.explained_variance_ratio_) / size(pca.explained_variance_ratio_))*100)
                print 'Compression Ratio %s %%' %(float(size(img_gray_pca)) / size(img_gray) * 100)
                show()

        else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return

    def Unsupervised(self,event):
         if current:
            src=current[0]

            # Output file name
            tgt = "classified.jpg"

            # Load the image into numpy using gdal
            srcArr = gdalnumeric.LoadFile(src)

            # Split the histogram into 20 bins as our classes
            classes = gdalnumeric.numpy.histogram(srcArr, bins=20)[1]

            # Color look-up table (LUT) - must be len(classes)+1.
            # Specified as R,G,B tuples 
            lut = [[255,0,0],[191,48,48],[166,0,0],[255,64,64],
            [255,115,115],[255,116,0],[191,113,48],[255,178,115],
            [0,153,153],[29,115,115],[0,99,99],[166,75,0],
            [0,204,0],[51,204,204],[255,150,64],[92,204,204],[38,153,38],
            [0,133,0],[57,230,57],[103,230,103],[184,138,0]]

            # Starting value for classification
            start = 1

            # Set up the RGB color JPEG output image
            rgb = gdalnumeric.numpy.zeros((3, srcArr.shape[0],
            srcArr.shape[1],), gdalnumeric.numpy.float32)
       
            # Process all classes and assign colors
            for i in range(len(classes)):
                mask = gdalnumeric.numpy.logical_and(start <= \
             srcArr, srcArr <= classes[i])
                for j in range(len(lut[i])):
                    rgb[j] = gdalnumeric.numpy.choose(mask, (rgb[j],lut[i][j])) 
    
                start = classes[i]+1 


            # Save the image    
            gdalnumeric.SaveArray(rgb.astype(gdalnumeric.numpy.uint8),tgt, format="JPEG")
            QtGui.QMessageBox.information(self,"Message","The classified image has been saved in your current directory")

         else:
            QtGui.QMessageBox.information(self,"Message","Please select an image first")
            return
             

    def About(self,event):
        QtGui.QMessageBox.about(self,"About ","<p>The Software <b> SATTELITE IMAGE MANIPULATOR </b>performs various operations on a digital image to enhance the visual appearance and to extract information from the image,the GUI shows various options using which images can be opened and analysed to perform various operations.</p>")


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
