from fundamentals.matlab_class import Struct
from fundamentals.utils import *
from parameters.basic_optical_parameters import basic_optical_parameters
import numpy as np
from numpy import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import skimage.restoration


"""
TIFF 파일 형식이 matlab code에서는 5D였는데(아마도 세로, 가로, RGB 3 channels) 그게 진짜인지
 -> 그렇다면 현재 2D input(세로, 가로) 가정한 코드를 수정해야함


"""


class Field_Retrieval(self):
    def __init__(self, init_params):
        self.parameters = Struct()
        self.utility = Struct()

        self.parameters = basic_optical_parameters() # Struct class

        # SIMULATION PARAMETERS
        self.parameters.resolution_image=np.array([1, 1])*0.1
        self.parameters.resolution = np.array([1,1])*0.1
        self.parameters.use_abbe_correction=True
        self.parameters.cutout_portion=1/3
        self.parameters.other_corner=False; #false if the field is in another corner of the image
        self.parameters.conjugate_field=False
        self.parameters.use_GPU = True
        
        if init_params is not None:
            self.parameters=update_struct(self.parameters, init_params)

    def check_input(h, input_field, output_field, ROI):
        if input_field.shape != output_field.shape:
            raise ValueError('Background and sample field must be of same size');
    
        if input_field.shape[0] != input_field.shape[1]:
            raise ValueError('the image must be a square')
        
        if h.parameters.resolution_image[0] != h.parameters.resolution_image[1]:
            raise ValueError('please enter an isotropic resolution for the image')
        
        if h.parameters.resolution[0] != h.parameters.resolution[1]:
            raise ValueError('please enter an isotropic resolution for the output image')
    
    def crop_rectangular_image(h, input_field, output_field, ROI):
        temp_input_field = fftshift(fft2(ifftshift(input_field)))
        temp_output_field = fftshift(fft2(ifftshift(output_field)))

        # 1 center the field in the fourier space
        delete_band_1 = [round(temp_input_field.shape[0]*h.parameters.cutout_portion), temp_input_field.shape[0]]
        delete_band_2 = [round(temp_input_field.shape[1]*h.parameters.cutout_portion), temp_input_field.shape[1]]
        if h.parameters.other_corner:
            delete_band_2=[1,round(input_field0.shape[1])*(1-h.parameters.cutout_portion)]

        normal_bg=temp_input_field
        normal_bg[delete_band_1[0]:delete_band_1[1], :] = 0
        normal_bg[:, delete_band_2[0]:delete_band_2[1]] = 0

        [center_pos_1,center_pos_2] = np.where(np.abs(normal_bg) == np.abs(normal_bg).max())

        input_field0=fftshift(fftshift(circshift(input_field0,[1-center_pos_1,1-center_pos_2,0])))
        output_field0=fftshift(fftshift(circshift(output_field0,[1-center_pos_1,1-center_pos_2,0])))

        # 2 match to the resolution
        resolution0 = h.parameters.resolution
        h.parameters.resolution[0] = h.parameters.resolution_image[0]
        h.parameters.resolution[1] = h.parameters.resolution_image[1]
        h.parameters.size[0] = input_field0.shape[0]
        h.parameters.size[1] = input_field0.shape[1]

        h.utility=derive_optical_tool(h.parameters)
        
        temp_input_field = temp_input_field*h.utility.NA_circle
        temp_output_field = temp_output_field*h.utility.NA_circle
        temp_input_field=fftshift(ifft2(ifftshift(temp_input_field)))
        temp_output_field=fftshift(ifft2(ifftshift(temp_output_field)))

        retPhase=np.angle(temp_output_field/temp_input_field)
        retPhase = skimage.restoration.unwrap_phase(retPhase)

        if h.parameters.conjugate_field:
            retPhase = -retPhase
        
        ################################
        # TODO : 이 밑에 구현하기
        retPhase = PhiShiftMS(retPhase,1,1)

        while True:
            close all
            figure, imagesc(retPhase, [0 max(retPhase(:))]), axis image, colormap gray, colorbar
            title('Choose square ROI')
            r = drawrectangle;
            ROI = r.Position;
            ROI = [round(ROI(2)) round(ROI(2))+round(ROI(3)) round(ROI(1)) round(ROI(1))+round(ROI(3)) ];
            close all
            figure, imagesc(max(squeeze(retPhase(ROI(1):ROI(2), ROI(3):ROI(4),:,:)),[],3), [0 max(retPhase(:))]), axis image, colormap gray
            satisfied = input('Satisfied? 1/0: ')
            if satisfied:
                close;
                break;
            end
        end
        ###############################

        input_field = input_field[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        output_field = output_field[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        h.parameters.resolution = resolution0
    
        return h, input_field, output_field, ROI
    
    def match_to_the_ratio(h, input_field, output_field):
        # (2) match to the resolution
        old_side_size = input_field.shape[0]
        resolution_ratio = h.parameters.resolition[0]/h.parameters.resolution_image[0]
        if resolution_ratio >= 1:
            # crop
            crop_size=round((1/2)*(input_field.shape[0]-input_field.shape[0]/resolution_ratio))
            input_field=input_field[1+crop_size:input_field.shape[0]-crop_size,1+crop_size:input_field.shape[1]-crop_size]
            output_field=output_field[1+crop_size:output_field.shape[0]-crop_size,1+crop_size:output_field.shape[1]-crop_size]
        
        if resolution_ratio < 1 :
            padd_size=round((1/2)*(input_field.shape[0]-input_field.shape[0]/resolution_ratio))

            ## TODO : inplement pararray
            input_field=padarray(input_field, [padd_size, padd_size], 'both')
            output_field=padarray(output_field, [padd_size, padd_size], 'both')

        h.parameters.resolution[0] = h.parameters.resolution_image[0] * old_side_size / input.shape_[0]
        h.parameters.resolution[1] = h.parameters.resolution_image[1] * old_side_size / input.shape_[1]

        return h, input_field, output_field
    
    def center_the_field_to_the_fourier_space(h, input_field, output_field):
        # Center the field in the fourier space
        delete_band_1 = [round(input_field.shape[0]*h.parameters.cutout_portion), round(input_field.shape[0])]
        delete_band_2 = [round(input_field.shape[1]*h.parameters.cutout_portion), round(input_field.shape[1])]
        if h.parameters.other_corner:
            delete_band_2 = [0, round(input_field.shape[1] * h.parameters.cutout_portion)]
        
        normal_bg = input_field
        normal_bg[delete_band_1[0]:delete_band_1[1], :] = 0
        normal_bg[:, delete_band_2[0]:delete_band_2[1]] = 0

        [center_pos_1,center_pos_2] = np.where(np.abs(normal_bg) == np.abs(normal_bg).max())
        input_field=fftshift(fftshift(circshift(input_field,[1-center_pos_1,1-center_pos_2,0])))
        output_field=fftshift(fftshift(circshift(output_field,[1-center_pos_1,1-center_pos_2,0])))
        
        return h, input_field, output_field

    def get_fields(self, bg_file, sp_file, ROI):
        h = self
        input_field=np.array(Image.open(bg_file))
        output_field=np.array(Image.open(sp_file))
        
        # Input Check
        self.check_input(h, input_field, output_field, ROI)

        if isinstance(ROI, list):
                input_field = input_field[ROI[0]:ROI[1], ROI[2]:ROI[3]]
                output_field = output_field[ROI[0]:ROI[1], ROI[2]:ROI[3]]         
        elif ROI=='rectangle' or ROI=='Rectangle':
            h, input_field, output_field, ROI = self.crop_rectangular_image(h, input_field, output_field, ROI)
        else:
            raise TypeError('Unknown ROI function')
        
        # size(input field)
        # size(output_field)

        is_overexposed = (max(max(input_field, output_field),[],[1,2])>254)

        input_field = fftshift(fft2(ifftshift(input_field)))
        output_field = fftshift(fft2(ifftshift(output_field)))

        # Center the field in the fourier space

        h, input_field, output_field = self.center_the_field_to_the_fourier_space(h, input_field, output_field)

        # (2) match to the resolution
        h, input_field, output_field = self.match_to_the_ratio(h, input_field, output_field)


        # crop the NA
        
        # TODO : implemnt warning('off', 'all')
        h.parameters.size[0] = input_field.shape[0]
        h.parameters.size[1] = input_field.shape[1]

        h.utility = derive_optical_tool(h.parameters)
        # warning('on', 'all')

        input_field = input_field * h.ulitily.NA_circle
        output_field = output_field * h.utility.NA_circle

        # Find peaks
        k0s = 